//! `ai-memory embed` — compute + store embeddings for every latest page.
//!
//! Used to backfill embeddings after attaching an embedder for the
//! first time, or after switching `(provider, model, dim)`.

use ai_memory_llm::{build_embedder, embedder_from_env};
use ai_memory_store::{Store, f32_vec_to_bytes};
use ai_memory_wiki::Wiki;
use anyhow::{Context, Result, bail};
use tracing::{info, warn};

use crate::cli::EmbedArgs;
use crate::config::Config;

/// Run the `embed` subcommand.
///
/// # Errors
/// Returns an error if no embedder is configured, the store cannot
/// be opened, or any embed/store call fails.
pub async fn run(config: &Config, args: EmbedArgs) -> Result<()> {
    let Some(cfg) = embedder_from_env()? else {
        bail!(
            "AI_MEMORY_EMBEDDING_PROVIDER is unset. Configure an embedder \
             (see CLAUDE.md / docs/design-decisions.md §4) and re-run."
        );
    };
    let provider = cfg.provider.name().to_string();
    let model = cfg.model.clone();
    let dim = cfg.dim;
    let embedder = build_embedder(cfg).context("building embedder")?;

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
    let wiki = Wiki::new(&config.data_dir, store.writer.clone())?;

    // Snapshot all current pages.
    let candidates = store.reader.decay_candidates(ws, proj).await?;
    info!(
        provider = embedder.provider(),
        model = embedder.model(),
        dim = embedder.dim(),
        candidates = candidates.len(),
        "starting embed backfill",
    );

    // Optimisation: when --force is off, skip pages that already have
    // a matching embedding row.
    let already: std::collections::HashSet<_> = if args.force {
        std::collections::HashSet::new()
    } else {
        store
            .reader
            .load_embeddings(ws, proj, provider.clone(), model.clone(), dim)
            .await?
            .into_iter()
            .map(|s| s.id)
            .collect()
    };

    let mut embedded = 0_usize;
    let mut skipped = 0_usize;
    let mut failed = 0_usize;
    for cand in candidates {
        if !args.force && already.contains(&cand.id) {
            skipped += 1;
            continue;
        }
        let md = match wiki.read_page(&cand.path) {
            Ok(m) => m,
            Err(e) => {
                warn!(path = %cand.path, error = %e, "skip: unable to read page");
                failed += 1;
                continue;
            }
        };
        if args.dry_run {
            embedded += 1;
            continue;
        }
        let vec = match embedder.embed(&md.body).await {
            Ok(v) => v,
            Err(e) => {
                warn!(path = %cand.path, error = %e, "embed call failed");
                failed += 1;
                continue;
            }
        };
        let bytes = f32_vec_to_bytes(&vec);
        if let Err(e) = store
            .writer
            .store_embedding(cand.id, bytes, provider.clone(), model.clone(), dim)
            .await
        {
            warn!(path = %cand.path, error = %e, "store_embedding failed");
            failed += 1;
            continue;
        }
        embedded += 1;
    }

    let report = serde_json::json!({
        "dry_run": args.dry_run,
        "provider": provider,
        "model": model,
        "dim": dim,
        "embedded": embedded,
        "skipped": skipped,
        "failed": failed,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
