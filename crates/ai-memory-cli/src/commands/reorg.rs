//! `ai-memory reorg --by-cwd [--dry-run]`
//!
//! Retro-fits existing sessions + observations to per-cwd projects based
//! on the `cwd` captured at session-start. Pages are graveyarded
//! (`is_latest=0`) because they were synthesised as a mash-up across all
//! sessions that used the old single-project bucket; a fresh consolidation
//! will regenerate them cleanly per project. The operation is idempotent:
//! sessions already in the right project are skipped.

use ai_memory_core::{ProjectId, SessionId, WorkspaceId};
use ai_memory_store::{Store, StoreError};
use anyhow::Result;
use tracing::info;

use crate::cli::ReorgArgs;
use crate::config::Config;

/// A single entry in the reorg plan.
#[derive(Debug)]
struct PlanEntry {
    session_id: SessionId,
    new_project_id: ProjectId,
    cwd: String,
    project_name: String,
}

/// Run the `reorg` subcommand.
///
/// # Errors
/// Returns an error if the store cannot be opened or if any SQL operation
/// fails.
pub async fn run(config: &Config, args: ReorgArgs) -> Result<()> {
    let store = Store::open(&config.data_dir)?;

    // Step 1: Ensure the `default` workspace exists (idempotent).
    let ws = store.writer.get_or_create_workspace("default").await?;

    // Step 2: Read all sessions with a non-NULL, non-empty cwd.
    let sessions_with_cwd: Vec<(SessionId, ProjectId, String)> = store
        .reader
        .with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, project_id, cwd \
                 FROM sessions \
                 WHERE cwd IS NOT NULL AND cwd != '' \
                 ORDER BY started_at",
            )?;
            let rows = stmt.query_map([], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let proj_bytes: Vec<u8> = row.get(1)?;
                let cwd: String = row.get(2)?;
                Ok((id_bytes, proj_bytes, cwd))
            })?;
            let mut out = Vec::new();
            for r in rows {
                let (id_bytes, proj_bytes, cwd) = r?;
                // from_slice returns MemoryError; lift through StoreError.
                let session_id = SessionId::from_slice(&id_bytes).map_err(StoreError::Memory)?;
                let project_id = ProjectId::from_slice(&proj_bytes).map_err(StoreError::Memory)?;
                out.push((session_id, project_id, cwd));
            }
            Ok(out)
        })
        .await?;

    if sessions_with_cwd.is_empty() {
        println!("No sessions with a cwd found; nothing to reorg.");
        return Ok(());
    }

    // Step 3: Derive project_name = basename(cwd) for each session and
    // resolve the target (workspace_id, project_id). Cache by cwd so we
    // call get_or_create_project once per distinct path.
    let mut cwd_to_proj: std::collections::HashMap<String, (WorkspaceId, ProjectId)> =
        std::collections::HashMap::new();
    for (_, _, cwd) in &sessions_with_cwd {
        if cwd_to_proj.contains_key(cwd.as_str()) {
            continue;
        }
        let project_name = std::path::Path::new(cwd.as_str())
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| "unknown".to_string());
        let proj = store
            .writer
            .get_or_create_project(ws, project_name, Some(cwd.clone()))
            .await?;
        cwd_to_proj.insert(cwd.clone(), (ws, proj));
    }

    // Step 4: Build the plan — only include sessions whose project_id
    // differs from what it should be (idempotency).
    let mut plan: Vec<PlanEntry> = Vec::new();
    for (session_id, old_project_id, cwd) in &sessions_with_cwd {
        let (_, new_project_id) = cwd_to_proj[cwd.as_str()];
        if new_project_id == *old_project_id {
            // Already in the right bucket.
            continue;
        }
        let project_name = std::path::Path::new(cwd.as_str())
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        plan.push(PlanEntry {
            session_id: *session_id,
            new_project_id,
            cwd: cwd.clone(),
            project_name,
        });
    }

    // Step 5: Dry-run: print the plan as a table and exit.
    if args.dry_run || plan.is_empty() {
        if plan.is_empty() {
            println!("All sessions are already in the correct per-cwd project; nothing to do.");
        } else {
            println!("{} session(s) would be moved:\n", plan.len());
            println!("{:<38} {:<20} cwd", "session_id", "new_project");
            println!("{}", "-".repeat(80));
            for e in &plan {
                println!(
                    "{:<38} {:<20} {}",
                    e.session_id.to_string(),
                    e.project_name,
                    e.cwd,
                );
            }
            println!("\n(dry-run; omit --dry-run to apply)");
        }
        return Ok(());
    }

    // Step 6: Execute in one transaction via the writer actor.
    let writer_plan: Vec<(SessionId, ProjectId)> = plan
        .iter()
        .map(|e| (e.session_id, e.new_project_id))
        .collect();
    let summary = store.writer.reorg_sessions(writer_plan).await?;

    // Count distinct new projects created (those with project_id not seen
    // in the old bucket, i.e. all entries in cwd_to_proj whose project_id
    // appears in the plan's new_project_id set).
    let distinct_projects: std::collections::HashSet<ProjectId> =
        plan.iter().map(|e| e.new_project_id).collect();

    info!(
        sessions_moved = summary.sessions_moved,
        observations_updated = summary.observations_updated,
        pages_graveyarded = summary.pages_graveyarded,
        distinct_projects = distinct_projects.len(),
        "reorg complete",
    );
    println!("Reorg complete:");
    println!("  sessions moved:        {}", summary.sessions_moved);
    println!("  observations updated:  {}", summary.observations_updated);
    println!("  pages graveyarded:     {}", summary.pages_graveyarded);
    println!("  distinct new projects: {}", distinct_projects.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory_core::{
        AgentKind, NewObservation, NewPage, NewSession, ObservationKind, PagePath, Tier,
    };
    use ai_memory_store::Store;
    use tempfile::TempDir;

    /// Seed a store with two sessions in two different cwds plus some
    /// observations and a page. After reorg the sessions + observations
    /// must land in per-cwd projects and the page must be graveyarded.
    #[tokio::test]
    async fn reorg_moves_sessions_and_graveyards_pages() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();

        // Create the initial "scratch" project (server default).
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let scratch = store
            .writer
            .get_or_create_project(ws, "scratch", None)
            .await
            .unwrap();

        // Session A — cwd=/home/user/project-alpha
        let sid_a = SessionId::new();
        store
            .writer
            .begin_session(NewSession {
                id: sid_a,
                workspace_id: ws,
                project_id: scratch,
                agent_kind: AgentKind::ClaudeCode,
                cwd: Some(std::path::PathBuf::from("/home/user/project-alpha")),
            })
            .await
            .unwrap();
        store
            .writer
            .insert_observation(NewObservation {
                session_id: sid_a,
                workspace_id: ws,
                project_id: scratch,
                kind: ObservationKind::UserPrompt,
                title: "hello alpha".into(),
                body: "".into(),
                importance: 5,
            })
            .await
            .unwrap();

        // Session B — cwd=/home/user/project-beta
        let sid_b = SessionId::new();
        store
            .writer
            .begin_session(NewSession {
                id: sid_b,
                workspace_id: ws,
                project_id: scratch,
                agent_kind: AgentKind::ClaudeCode,
                cwd: Some(std::path::PathBuf::from("/home/user/project-beta")),
            })
            .await
            .unwrap();
        store
            .writer
            .insert_observation(NewObservation {
                session_id: sid_b,
                workspace_id: ws,
                project_id: scratch,
                kind: ObservationKind::UserPrompt,
                title: "hello beta".into(),
                body: "".into(),
                importance: 5,
            })
            .await
            .unwrap();

        // Insert a mash-up page in the scratch project.
        store
            .writer
            .upsert_page(NewPage {
                workspace_id: ws,
                project_id: scratch,
                path: PagePath::new("sessions/mash.md").unwrap(),
                title: "Mash-up page".into(),
                body: "multi-project content".into(),
                tier: Tier::Episodic,
                frontmatter_json: serde_json::json!({}),
                pinned: false,
            })
            .await
            .unwrap();

        // Verify baseline: one is_latest page.
        let counts_before = store.reader.status_counts().await.unwrap();
        assert_eq!(counts_before.pages_latest, 1, "one page before reorg");

        // Run reorg logic (not via CLI; call helpers directly).
        let ws_id = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();

        // Simulate what `run` does: read sessions, derive project names.
        let sessions_with_cwd: Vec<(SessionId, ProjectId, String)> = store
            .reader
            .with_conn(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, project_id, cwd FROM sessions \
                     WHERE cwd IS NOT NULL AND cwd != '' ORDER BY started_at",
                )?;
                let rows = stmt.query_map([], |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let proj_bytes: Vec<u8> = row.get(1)?;
                    let cwd: String = row.get(2)?;
                    Ok((id_bytes, proj_bytes, cwd))
                })?;
                let mut out = Vec::new();
                for r in rows {
                    let (id_bytes, proj_bytes, cwd) = r.unwrap();
                    let sid = SessionId::from_slice(&id_bytes).unwrap();
                    let pid = ProjectId::from_slice(&proj_bytes).unwrap();
                    out.push((sid, pid, cwd));
                }
                Ok(out)
            })
            .await
            .unwrap();

        assert_eq!(sessions_with_cwd.len(), 2);

        let mut cwd_to_proj: std::collections::HashMap<String, ProjectId> =
            std::collections::HashMap::new();
        for (_, _, cwd) in &sessions_with_cwd {
            if cwd_to_proj.contains_key(cwd.as_str()) {
                continue;
            }
            let name = std::path::Path::new(cwd.as_str())
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            let proj = store
                .writer
                .get_or_create_project(ws_id, name, Some(cwd.clone()))
                .await
                .unwrap();
            cwd_to_proj.insert(cwd.clone(), proj);
        }

        let plan: Vec<(SessionId, ProjectId)> = sessions_with_cwd
            .iter()
            .filter_map(|(sid, old_pid, cwd)| {
                let new_pid = cwd_to_proj[cwd.as_str()];
                if new_pid != *old_pid {
                    Some((*sid, new_pid))
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(plan.len(), 2, "both sessions need moving");

        let summary = store.writer.reorg_sessions(plan).await.unwrap();
        assert_eq!(summary.sessions_moved, 2);
        assert_eq!(summary.observations_updated, 2);
        assert_eq!(summary.pages_graveyarded, 1);

        // Pages should all be graveyarded (is_latest = 0).
        let counts_after = store.reader.status_counts().await.unwrap();
        assert_eq!(counts_after.pages_latest, 0, "page must be graveyarded");

        // Sessions should now be in distinct projects.
        let alpha_proj = cwd_to_proj["/home/user/project-alpha"];
        let beta_proj = cwd_to_proj["/home/user/project-beta"];
        assert_ne!(alpha_proj, beta_proj);

        // Re-read sessions from DB and verify they carry the correct project.
        // We piggy-back on the same read-sessions logic used by `run`.
        let sessions_after: Vec<(SessionId, ProjectId, String)> = store
            .reader
            .with_conn(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT id, project_id, cwd FROM sessions \
                     WHERE cwd IS NOT NULL AND cwd != '' ORDER BY started_at",
                )?;
                let rows = stmt.query_map([], |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let proj_bytes: Vec<u8> = row.get(1)?;
                    let cwd: String = row.get(2)?;
                    Ok((id_bytes, proj_bytes, cwd))
                })?;
                let mut out = Vec::new();
                for r in rows {
                    let (id_bytes, proj_bytes, cwd) = r?;
                    let sid = SessionId::from_slice(&id_bytes).map_err(StoreError::Memory)?;
                    let pid = ProjectId::from_slice(&proj_bytes).map_err(StoreError::Memory)?;
                    out.push((sid, pid, cwd));
                }
                Ok(out)
            })
            .await
            .unwrap();

        for (sid, pid, cwd) in &sessions_after {
            let expected = cwd_to_proj[cwd.as_str()];
            assert_eq!(
                *pid, expected,
                "session {sid} in cwd {cwd} should have project {expected:?}"
            );
        }
        assert!(
            sessions_after.iter().any(|(sid, _, _)| *sid == sid_a),
            "session A present"
        );
        assert!(
            sessions_after.iter().any(|(sid, _, _)| *sid == sid_b),
            "session B present"
        );
    }
}
