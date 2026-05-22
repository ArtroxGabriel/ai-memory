//! `ai-memory serve` — MCP server with optional filesystem watcher.

use ai_memory_hooks::{HookState, hook_router};
use ai_memory_llm::{build_embedder, embedder_from_env, provider_from_env};
use ai_memory_mcp::AiMemoryServer;
use ai_memory_store::Store;
use ai_memory_wiki::{WatcherHandle, Wiki};
use anyhow::{Context, Result};
use rmcp::ServiceExt;
use rmcp::transport::stdio;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::cli::{ServeArgs, TransportKind};
use crate::config::Config;

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
    let mut wiki = Wiki::new(&config.data_dir, store.writer.clone())?;

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
        .with_decay_params(config.decay);
    if let Some(e) = embedder.clone() {
        server = server.with_embedder(e);
    }
    if let Some(cfg) = provider_from_env()? {
        let llm = ai_memory_llm::build_provider(cfg).context("building LLM provider from env")?;
        info!(
            provider = llm.name(),
            model = llm.model(),
            "memory_consolidate + memory_lint LLM features enabled",
        );
        server = server.with_consolidator(wiki.clone(), llm);
    } else {
        info!(
            "AI_MEMORY_LLM_PROVIDER unset; memory_consolidate disabled, lint runs rule-based only"
        );
    }

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
            let mcp_service = StreamableHttpService::new(
                move || Ok(server_clone.clone()),
                LocalSessionManager::default().into(),
                StreamableHttpServerConfig::default().with_cancellation_token(cancel.child_token()),
            );
            let hooks = hook_router(HookState {
                workspace_id: ws,
                project_id: proj,
                writer: store.writer.clone(),
                reader: store.reader.clone(),
                wiki: wiki.clone(),
            });
            let router = axum::Router::new()
                .nest_service("/mcp", mcp_service)
                .merge(hooks);
            let listener = tokio::net::TcpListener::bind(&bind)
                .await
                .with_context(|| format!("binding {bind}"))?;
            info!(
                %bind,
                "MCP HTTP server ready (POST /mcp, POST /hook, Ctrl-C to stop)",
            );
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
