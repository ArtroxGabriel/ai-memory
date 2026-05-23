//! `ai-memory serve` — MCP server with optional filesystem watcher.

use std::sync::Arc;

use ai_memory_consolidate::Consolidator;
use ai_memory_core::Sanitizer;
use ai_memory_hooks::{HookState, hook_router};
use ai_memory_llm::{build_embedder, embedder_from_env, provider_from_env};
use ai_memory_mcp::AiMemoryServer;
use ai_memory_store::Store;
use ai_memory_web;
use ai_memory_wiki::{WatcherHandle, Wiki};
use anyhow::{Context, Result};
use axum::extract::DefaultBodyLimit;
use rmcp::ServiceExt;
use rmcp::transport::stdio;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::auth::{AuthState, require_bearer};
use crate::cli::{ServeArgs, TransportKind};
use crate::config::Config;

/// 10 MB cap on inbound HTTP bodies. The /hook ingress accepts the
/// agent's raw payload which can include a tool output excerpt
/// (capped at 2 KB on our side via `truncate_excerpt`), but Claude
/// Code et al. send the full envelope, which can run to a few KB.
/// 10 MB is generous headroom; without a cap, axum streams unbounded
/// bodies into memory (audit critical #2).
const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;

/// Run the `serve` subcommand.
///
/// # Errors
/// Returns an error if the store cannot be opened, the watcher cannot
/// install, or the transport setup fails.
pub async fn run(config: &Config, args: ServeArgs) -> Result<()> {
    let store = Store::open(&config.data_dir)
        .with_context(|| format!("opening store at {}", config.data_dir.display()))?;
    let ws = store
        .writer
        .get_or_create_workspace(args.workspace.clone())
        .await?;
    let proj = store
        .writer
        .get_or_create_project(ws, args.project.clone(), None)
        .await?;
    // Build the privacy strip from config. Compile errors in
    // user-supplied regex abort startup with a clear message so
    // operators discover misconfiguration immediately.
    let sanitizer = Sanitizer::new(&config.sanitize)
        .context("compiling sanitizer.extra_patterns from config")?;
    let mut wiki =
        Wiki::new(&config.data_dir, store.writer.clone())?.with_sanitizer(sanitizer.clone());

    // M9 — pluggable embedder. Refuse to start if any stored
    // embeddings disagree with the configured (provider, model, dim).
    let embedder = if let Some(cfg) = embedder_from_env()? {
        let mismatch = store
            .reader
            .embedding_meta_for_mismatch(cfg.provider.name().into(), cfg.model.clone(), cfg.dim)
            .await?;
        if !mismatch.is_empty() {
            anyhow::bail!(
                "embedding (provider, model, dim) mismatch with stored data: {:?} \
                 — run `ai-memory embed --reembed` to migrate",
                mismatch
            );
        }
        let e = build_embedder(cfg).context("building embedder from env")?;
        info!(
            provider = e.provider(),
            model = e.model(),
            dim = e.dim(),
            "embedder enabled"
        );
        wiki = wiki.with_embedder(e.clone());
        Some(e)
    } else {
        info!("AI_MEMORY_EMBEDDING_PROVIDER unset; hybrid search disabled (FTS5-only)");
        None
    };

    // Keep the guard alive for the lifetime of `serve`.
    let _watcher = if args.no_watcher {
        info!("watcher disabled by --no-watcher");
        None
    } else {
        info!(
            root = %wiki.root().display(),
            workspace = %args.workspace,
            project = %args.project,
            "starting wiki watcher",
        );
        Some(WatcherHandle::start(wiki.clone(), ws, proj)?)
    };

    let mut server = AiMemoryServer::new(store.reader.clone(), store.writer.clone(), ws, proj)
        .with_wiki(wiki.clone())
        .with_decay_params(config.decay)
        .with_sanitizer(sanitizer.clone());
    if let Some(e) = embedder.clone() {
        server = server.with_embedder(e);
    }
    // Build the consolidator (if LLM configured) once, then share the
    // Arc between the MCP server (for `memory_consolidate` + lint) and
    // the hook router (for PreCompact checkpointing).
    let consolidator: Option<Arc<Consolidator>> = if let Some(cfg) = provider_from_env()? {
        let llm = ai_memory_llm::build_provider(cfg).context("building LLM provider from env")?;
        info!(
            provider = llm.name(),
            model = llm.model(),
            "memory_consolidate + PreCompact LLM checkpointing enabled",
        );
        let c = Arc::new(Consolidator::new(
            store.reader.clone(),
            store.writer.clone(),
            wiki.clone(),
            llm.clone(),
            ws,
            proj,
        ));
        server = server.with_consolidator_arc(wiki.clone(), llm, c.clone());
        Some(c)
    } else {
        info!(
            "AI_MEMORY_LLM_PROVIDER unset; memory_consolidate disabled, PreCompact \
             falls back to rule-based checkpoint, lint runs rule-based only"
        );
        None
    };

    match args.transport {
        TransportKind::Stdio => {
            info!("MCP server ready on stdio (Ctrl-C to stop)");
            let service = server.serve(stdio()).await?;
            service.waiting().await?;
        }
        TransportKind::Http => {
            let bind = args.bind.unwrap_or_else(|| config.bind.clone());
            let cancel = CancellationToken::new();
            let server_clone = server.clone();
            // `Host`-header allowlist for rmcp's DNS-rebinding guard.
            // Sourced from Config (which already handles the
            // `AI_MEMORY_ALLOWED_HOSTS=a,b,c` env-string vs.
            // config.toml sequence forms via the string-or-vec
            // deserializer). Logged so operators can verify the
            // effective list against what they intended.
            info!(
                allowed_hosts = ?config.allowed_hosts,
                "MCP Host-header allowlist"
            );
            let mcp_service = StreamableHttpService::new(
                move || Ok(server_clone.clone()),
                LocalSessionManager::default().into(),
                StreamableHttpServerConfig::default()
                    .with_cancellation_token(cancel.child_token())
                    .with_allowed_hosts(config.allowed_hosts.clone()),
            );
            let hooks = hook_router(HookState {
                workspace_id: ws,
                project_id: proj,
                writer: store.writer.clone(),
                reader: store.reader.clone(),
                wiki: wiki.clone(),
                consolidator: consolidator.clone(),
                sanitizer: sanitizer.clone(),
                project_cache: std::sync::Arc::new(tokio::sync::Mutex::new(
                    std::collections::HashMap::new(),
                )),
            });
            // Build the auth state. Precedence (highest first):
            //   1. AI_MEMORY_AUTH_TOKEN env var
            //   2. config.toml [auth].bearer_token
            //   3. neither → open mode (no auth)
            // Read env directly (not via figment) to match the pattern
            // used by AI_MEMORY_LLM_* in factory.rs — keeps the
            // operator's mental model simple.
            let token = std::env::var("AI_MEMORY_AUTH_TOKEN")
                .ok()
                .filter(|s| !s.is_empty())
                .or_else(|| config.auth.bearer_token.clone());
            let auth_state = Arc::new(AuthState::new(token));
            let auth_enabled = auth_state.enabled();
            let mut router = axum::Router::new()
                .nest_service("/mcp", mcp_service)
                .merge(hooks);

            // Register the web router BEFORE applying the bearer
            // middleware. In axum 0.8, `.layer()` only attaches to
            // routes registered before the call — nesting after the
            // layer would silently bypass auth for /web/*.
            if args.enable_web {
                let web_router = ai_memory_web::router(store.reader.clone(), wiki.clone());
                // Also accept the trailing-slash form. axum 0.8's
                // `nest("/web", ..)` matches `/web` (no slash) → inner
                // `route("/")` but doesn't match `/web/` (a browser's
                // default when the user types the bare prefix), so we
                // redirect that explicitly to keep both URLs working.
                router = router
                    .route(
                        "/web/",
                        axum::routing::get(|| async {
                            axum::response::Redirect::permanent("/web")
                        }),
                    )
                    .nest("/web", web_router);
                info!("read-only wiki browser mounted at /web");
            }

            let router = router
                .layer(axum::middleware::from_fn_with_state(
                    auth_state,
                    require_bearer,
                ))
                .layer(DefaultBodyLimit::max(MAX_BODY_BYTES));
            let listener = tokio::net::TcpListener::bind(&bind)
                .await
                .with_context(|| format!("binding {bind}"))?;
            info!(
                %bind,
                auth = auth_enabled,
                body_limit_mb = MAX_BODY_BYTES / 1024 / 1024,
                "MCP HTTP server ready (POST /mcp, POST /hook, Ctrl-C to stop)",
            );
            if !auth_enabled && !bind.starts_with("127.") {
                // Loud warning: a non-loopback bind with no auth is
                // the audit's critical-#1 scenario. The operator gets
                // a one-line "you sure?" instead of silent exposure.
                tracing::warn!(
                    %bind,
                    "no AI_MEMORY_AUTH_TOKEN configured AND binding to a non-loopback \
                     address — anyone on the network can call destructive MCP tools. \
                     Generate a token with `ai-memory generate-auth-token` and set \
                     AI_MEMORY_AUTH_TOKEN in the server's environment."
                );
            }
            axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    let _ = tokio::signal::ctrl_c().await;
                    info!("ctrl-c received; shutting down");
                    cancel.cancel();
                })
                .await?;
        }
    }
    Ok(())
}
