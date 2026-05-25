//! Mutating SQL operations executed on the writer thread.
//!
//! Each operation is one transaction. Calling them from anywhere other than
//! the writer thread would violate the single-writer invariant (see
//! [`crate::writer`]).

use std::collections::BTreeSet;

use ai_memory_core::{
    AgentKind, HandoffId, NewHandoff, NewObservation, NewPage, NewSession, ObservationId,
    ObservationKind, PageId, PagePath, ProjectId, SessionId, WorkspaceId,
};

/// Summary returned by [`reorg_sessions`] and exposed via
/// [`crate::writer::WriterHandle::reorg_sessions`].
#[derive(Debug, Default, Clone)]
pub struct ReorgSummary {
    /// Sessions whose `project_id` was changed.
    pub sessions_moved: usize,
    /// Observations updated to match their session's new project.
    pub observations_updated: usize,
    /// `is_latest=1` pages marked `is_latest=0` (mash-up graveyard).
    pub pages_graveyarded: usize,
}

/// Summary returned by [`purge_project`] and exposed via
/// [`crate::writer::WriterHandle::purge_project`].
#[derive(Debug, Default, Clone)]
pub struct PurgeSummary {
    /// Human-readable `workspace/project` label. Set by the caller (writer
    /// only knows IDs); filled in by [`purge_project`] from its parameters.
    pub label: String,
    /// Distinct page paths that were present before the delete (all versions,
    /// not just `is_latest=1`). The admin handler uses this list to remove
    /// the corresponding files from the wiki directory.
    pub page_paths: Vec<String>,
    /// Number of `pages` rows deleted (all versions, not just latest).
    pub pages_deleted: u64,
    /// Number of `sessions` rows deleted.
    pub sessions_deleted: u64,
    /// Number of `observations` rows deleted.
    pub observations_deleted: u64,
    /// Number of `handoffs` rows deleted.
    pub handoffs_deleted: u64,
    /// Number of `page_embeddings` rows deleted (cascades through pages).
    pub embeddings_deleted: u64,
}
use jiff::Timestamp;
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::error::{StoreError, StoreResult};

/// One embedding upsert requested by a backfill or embed command.
#[derive(Debug)]
pub struct EmbeddingWrite {
    /// Page receiving the embedding.
    pub page_id: PageId,
    /// Packed little-endian `f32` vector bytes.
    pub vector_bytes: Vec<u8>,
    /// Embedding provider name.
    pub provider: String,
    /// Embedding model name.
    pub model: String,
    /// Vector dimension.
    pub dim: u32,
}

/// Upsert a page by path, superseding any existing latest version when the
/// content (sha256 of body) has changed.
///
/// Returns the id of the page row that should now be considered current.
pub fn upsert_page(conn: &mut Connection, page: &NewPage) -> StoreResult<PageId> {
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    let result_id = upsert_page_in_tx(&tx, page, now)?;
    tx.commit()?;
    Ok(result_id)
}

/// Resolve a workspace by name, creating it if missing. Atomic.
pub fn get_or_create_workspace(
    conn: &mut Connection,
    name: &str,
) -> StoreResult<ai_memory_core::WorkspaceId> {
    let tx = conn.transaction()?;
    let existing: Option<Vec<u8>> = tx
        .query_row(
            "SELECT id FROM workspaces WHERE name = ?1",
            params![name],
            |row| row.get(0),
        )
        .optional()?;
    let id = if let Some(bytes) = existing {
        ai_memory_core::WorkspaceId::from_slice(&bytes)?
    } else {
        let id = ai_memory_core::WorkspaceId::new();
        tx.execute(
            "INSERT INTO workspaces (id, name, created_at) VALUES (?1, ?2, ?3)",
            params![id.as_bytes(), name, Timestamp::now().as_microsecond()],
        )?;
        id
    };
    tx.commit()?;
    Ok(id)
}

/// Resolve a project by `(workspace_id, name)`, creating it if missing.
/// Atomic.
pub fn get_or_create_project(
    conn: &mut Connection,
    workspace_id: &ai_memory_core::WorkspaceId,
    name: &str,
    repo_path: Option<&str>,
) -> StoreResult<ai_memory_core::ProjectId> {
    let tx = conn.transaction()?;
    let existing: Option<Vec<u8>> = tx
        .query_row(
            "SELECT id FROM projects WHERE workspace_id = ?1 AND name = ?2",
            params![workspace_id.as_bytes(), name],
            |row| row.get(0),
        )
        .optional()?;
    let id = if let Some(bytes) = existing {
        ai_memory_core::ProjectId::from_slice(&bytes)?
    } else {
        let id = ai_memory_core::ProjectId::new();
        tx.execute(
            "INSERT INTO projects (id, workspace_id, name, repo_path, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                id.as_bytes(),
                workspace_id.as_bytes(),
                name,
                repo_path,
                Timestamp::now().as_microsecond()
            ],
        )?;
        id
    };
    tx.commit()?;
    Ok(id)
}

/// Upsert a batch of pages inside one transaction. Either *all* pages
/// land (each becoming the new `is_latest=true` version) or none do.
///
/// This is the M7b atomic-fan-out path: the consolidator can hand a
/// list of {sessions, concepts, decisions} pages and trust that
/// either the whole batch supersedes or the wiki is unchanged.
pub fn upsert_pages_batch(conn: &mut Connection, pages: &[NewPage]) -> StoreResult<Vec<PageId>> {
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    let mut out = Vec::with_capacity(pages.len());
    for page in pages {
        let id = upsert_page_in_tx(&tx, page, now)?;
        out.push(id);
    }
    tx.commit()?;
    Ok(out)
}

fn upsert_page_in_tx(
    tx: &rusqlite::Transaction<'_>,
    page: &NewPage,
    now: i64,
) -> StoreResult<PageId> {
    let body_sha256: [u8; 32] = {
        let mut hasher = Sha256::new();
        hasher.update(page.body.as_bytes());
        hasher.finalize().into()
    };
    let frontmatter_str = serde_json::to_string(&page.frontmatter_json)?;
    let tier_str = page.tier.as_str();

    let existing: Option<(Vec<u8>, Vec<u8>)> = tx
        .query_row(
            "SELECT id, body_sha256 FROM pages \
             WHERE workspace_id = ?1 AND project_id = ?2 AND path = ?3 AND is_latest = 1",
            params![
                page.workspace_id.as_bytes(),
                page.project_id.as_bytes(),
                page.path.as_str(),
            ],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;

    if let Some((existing_id, existing_sha)) = existing {
        if existing_sha == body_sha256 {
            return PageId::from_slice(&existing_id).map_err(StoreError::from);
        }
        let new_id = PageId::new();
        tx.execute(
            "UPDATE pages SET is_latest = 0 WHERE id = ?1",
            params![existing_id],
        )?;
        tx.execute(
            "INSERT INTO pages \
             (id, workspace_id, project_id, path, title, tier, body, body_sha256, \
              frontmatter_json, is_latest, supersedes, pinned, created_at, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1, ?10, ?11, ?12, ?12)",
            params![
                new_id.as_bytes(),
                page.workspace_id.as_bytes(),
                page.project_id.as_bytes(),
                page.path.as_str(),
                page.title,
                tier_str,
                page.body,
                body_sha256.as_slice(),
                frontmatter_str,
                existing_id,
                i64::from(page.pinned),
                now,
            ],
        )?;
        replace_links_in_tx(tx, &new_id, page)?;
        refresh_incoming_links_for_path(tx, page, &new_id)?;
        audit(
            tx,
            "supersede_page",
            Some(page.workspace_id.as_bytes()),
            Some(page.project_id.as_bytes()),
            Some(new_id.as_bytes()),
            now,
        )?;
        return Ok(new_id);
    }
    let new_id = PageId::new();
    tx.execute(
        "INSERT INTO pages \
         (id, workspace_id, project_id, path, title, tier, body, body_sha256, \
          frontmatter_json, is_latest, pinned, created_at, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1, ?10, ?11, ?11)",
        params![
            new_id.as_bytes(),
            page.workspace_id.as_bytes(),
            page.project_id.as_bytes(),
            page.path.as_str(),
            page.title,
            tier_str,
            page.body,
            body_sha256.as_slice(),
            frontmatter_str,
            i64::from(page.pinned),
            now,
        ],
    )?;
    replace_links_in_tx(tx, &new_id, page)?;
    refresh_incoming_links_for_path(tx, page, &new_id)?;
    audit(
        tx,
        "create_page",
        Some(page.workspace_id.as_bytes()),
        Some(page.project_id.as_bytes()),
        Some(new_id.as_bytes()),
        now,
    )?;
    Ok(new_id)
}

fn replace_links_in_tx(
    tx: &rusqlite::Transaction<'_>,
    from_page_id: &PageId,
    page: &NewPage,
) -> StoreResult<()> {
    tx.execute(
        "DELETE FROM links WHERE from_page_id = ?1",
        params![from_page_id.as_bytes()],
    )?;

    let mut seen = BTreeSet::new();
    for to_path in &page.links {
        if !seen.insert(to_path.as_str().to_string()) {
            continue;
        }
        let to_page_id = latest_page_id_for_path(tx, page, to_path)?;
        let to_page_blob = to_page_id.as_ref().map(|id| &id.as_bytes()[..]);
        tx.execute(
            "INSERT INTO links (from_page_id, to_page_id, to_path, link_type) \
             VALUES (?1, ?2, ?3, 'references')",
            params![from_page_id.as_bytes(), to_page_blob, to_path.as_str()],
        )?;
    }
    Ok(())
}

fn latest_page_id_for_path(
    tx: &rusqlite::Transaction<'_>,
    page: &NewPage,
    to_path: &PagePath,
) -> StoreResult<Option<PageId>> {
    let bytes: Option<Vec<u8>> = tx
        .query_row(
            "SELECT id FROM pages \
             WHERE workspace_id = ?1 AND project_id = ?2 AND path = ?3 AND is_latest = 1",
            params![
                page.workspace_id.as_bytes(),
                page.project_id.as_bytes(),
                to_path.as_str(),
            ],
            |row| row.get(0),
        )
        .optional()?;
    bytes
        .map(|bytes| PageId::from_slice(&bytes).map_err(StoreError::from))
        .transpose()
}

fn refresh_incoming_links_for_path(
    tx: &rusqlite::Transaction<'_>,
    page: &NewPage,
    latest_page_id: &PageId,
) -> StoreResult<()> {
    tx.execute(
        "UPDATE links \
         SET to_page_id = ?1 \
         WHERE to_path = ?2 \
           AND EXISTS ( \
               SELECT 1 FROM pages from_page \
               WHERE from_page.id = links.from_page_id \
                 AND from_page.workspace_id = ?3 \
                 AND from_page.project_id = ?4 \
           )",
        params![
            latest_page_id.as_bytes(),
            page.path.as_str(),
            page.workspace_id.as_bytes(),
            page.project_id.as_bytes(),
        ],
    )?;
    Ok(())
}

/// Begin (or re-affirm) a session row keyed on the caller-supplied id.
/// Idempotent: a second call with the same id leaves the row untouched.
pub fn begin_session(conn: &mut Connection, session: &NewSession) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let agent = session.agent_kind.as_str();
    let cwd: Option<String> = session
        .cwd
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    conn.execute(
        "INSERT INTO sessions (id, workspace_id, project_id, agent_kind, cwd, started_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(id) DO NOTHING",
        params![
            session.id.as_bytes(),
            session.workspace_id.as_bytes(),
            session.project_id.as_bytes(),
            agent,
            cwd,
            now,
        ],
    )?;
    Ok(())
}

/// Stamp a session as ended, optionally linking the synthesised summary
/// page.
pub fn end_session(
    conn: &mut Connection,
    session_id: &SessionId,
    summary_page_id: Option<&PageId>,
) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let page_blob: Option<&[u8]> = summary_page_id.map(|p| &p.as_bytes()[..]);
    conn.execute(
        "UPDATE sessions SET ended_at = ?1, summary_page_id = ?2 WHERE id = ?3",
        params![now, page_blob, session_id.as_bytes()],
    )?;
    Ok(())
}

/// Append a single observation. Caller is expected to have already
/// inserted the parent session via [`begin_session`].
pub fn insert_observation(
    conn: &mut Connection,
    obs: &NewObservation,
) -> StoreResult<ObservationId> {
    let id = ObservationId::new();
    let now = Timestamp::now().as_microsecond();
    let kind = observation_kind_as_str(obs.kind);
    let importance: i64 = i64::from(obs.importance.clamp(1, 10));
    conn.execute(
        "INSERT INTO observations \
         (id, session_id, workspace_id, project_id, kind, title, body, importance, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            id.as_bytes(),
            obs.session_id.as_bytes(),
            obs.workspace_id.as_bytes(),
            obs.project_id.as_bytes(),
            kind,
            obs.title,
            obs.body,
            importance,
            now,
        ],
    )?;
    Ok(id)
}

/// Store / replace one page's embedding. Bytes are the host-endian
/// `f32` packing of the unit-normalised vector. Provider/model/dim
/// are denormalised onto the row so a single SELECT can detect
/// heterogeneity (refuse-on-mismatch path).
pub fn store_embedding(
    conn: &mut Connection,
    page_id: &PageId,
    vector_bytes: &[u8],
    provider: &str,
    model: &str,
    dim: u32,
) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    conn.execute(
        "INSERT INTO page_embeddings (page_id, vector, provider, model, dim, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
         ON CONFLICT(page_id) DO UPDATE SET \
             vector = excluded.vector, \
             provider = excluded.provider, \
             model = excluded.model, \
             dim = excluded.dim, \
             created_at = excluded.created_at",
        params![page_id.as_bytes(), vector_bytes, provider, model, dim, now,],
    )?;
    Ok(())
}

/// Store / replace a batch of page embeddings in one transaction.
pub fn store_embeddings(conn: &mut Connection, embeddings: &[EmbeddingWrite]) -> StoreResult<()> {
    if embeddings.is_empty() {
        return Ok(());
    }
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO page_embeddings (page_id, vector, provider, model, dim, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) \
             ON CONFLICT(page_id) DO UPDATE SET \
                 vector = excluded.vector, \
                 provider = excluded.provider, \
                 model = excluded.model, \
                 dim = excluded.dim, \
                 created_at = excluded.created_at",
        )?;
        for embedding in embeddings {
            stmt.execute(params![
                embedding.page_id.as_bytes(),
                embedding.vector_bytes.as_slice(),
                embedding.provider.as_str(),
                embedding.model.as_str(),
                embedding.dim,
                now,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Bump `access_count` + `last_accessed_at` for the pages whose ids
/// appear in `page_ids`. Idempotent for unknown ids (no-op).
/// Used by the read path to feed the M8 reinforcement term.
pub fn bump_access_for_pages(conn: &mut Connection, page_ids: &[PageId]) -> StoreResult<()> {
    if page_ids.is_empty() {
        return Ok(());
    }
    let now = Timestamp::now().as_microsecond();
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "UPDATE pages \
             SET access_count = access_count + 1, last_accessed_at = ?1 \
             WHERE id = ?2 AND is_latest = 1",
        )?;
        for id in page_ids {
            stmt.execute(params![now, id.as_bytes()])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Mark a set of `is_latest=1` pages as soft-deleted by the forget
/// sweep. Distinguished from M7 supersession by `supersedes IS NULL`.
pub fn soft_delete_for_decay(conn: &mut Connection, page_ids: &[PageId]) -> StoreResult<usize> {
    if page_ids.is_empty() {
        return Ok(0);
    }
    let now = Timestamp::now().as_microsecond();
    let mut affected = 0usize;
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "UPDATE pages \
             SET is_latest = 0, superseded_at = ?1 \
             WHERE id = ?2 AND is_latest = 1",
        )?;
        for id in page_ids {
            affected += stmt.execute(params![now, id.as_bytes()])?;
        }
    }
    audit(
        &tx,
        "soft_delete_for_decay",
        None,
        None,
        None,
        Timestamp::now().as_microsecond(),
    )?;
    tx.commit()?;
    Ok(affected)
}

/// Hard-delete rows that were soft-deleted by an earlier sweep at
/// least `hard_delete_after_days` ago AND received zero subsequent
/// accesses. Safe: M7 supersedes-chain pages have a non-null
/// `supersedes` so they never match.
pub fn hard_delete_decayed_pages(
    conn: &mut Connection,
    hard_delete_after_days: i64,
) -> StoreResult<usize> {
    let cutoff = Timestamp::now().as_microsecond() - hard_delete_after_days * 86_400_000_000;
    let n = conn.execute(
        "DELETE FROM pages \
         WHERE is_latest = 0 \
           AND supersedes IS NULL \
           AND superseded_at IS NOT NULL \
           AND superseded_at < ?1 \
           AND access_count = 0",
        params![cutoff],
    )?;
    Ok(n)
}

/// Insert a new handoff in state=open.
pub fn insert_handoff(conn: &mut Connection, h: &NewHandoff) -> StoreResult<HandoffId> {
    let id = HandoffId::new();
    let now = Timestamp::now().as_microsecond();
    let open_q = serde_json::to_string(&h.open_questions)?;
    let next_s = serde_json::to_string(&h.next_steps)?;
    let files = serde_json::to_string(&h.files_touched)?;
    let from_session: Option<&[u8]> = h.from_session_id.as_ref().map(|s| &s.as_bytes()[..]);
    let cwd: Option<String> = h.cwd.as_ref().map(|p| p.to_string_lossy().into_owned());
    let from_agent = h.from_agent.as_str();
    let to_agent = h.to_agent.map(AgentKind::as_str);
    conn.execute(
        "INSERT INTO handoffs \
         (id, workspace_id, project_id, from_session_id, from_agent, to_agent, cwd, summary, \
          open_questions, next_steps, files_touched, state, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'open', ?12)",
        params![
            id.as_bytes(),
            h.workspace_id.as_bytes(),
            h.project_id.as_bytes(),
            from_session,
            from_agent,
            to_agent,
            cwd,
            h.summary,
            open_q,
            next_s,
            files,
            now,
        ],
    )?;
    Ok(id)
}

/// Mark a handoff accepted by `accepting_agent` / `accepting_session`.
pub fn accept_handoff(
    conn: &mut Connection,
    handoff_id: &HandoffId,
    accepting_agent: AgentKind,
    accepting_session: Option<&SessionId>,
) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let agent = accepting_agent.as_str();
    let session: Option<&[u8]> = accepting_session.map(|s| &s.as_bytes()[..]);
    conn.execute(
        "UPDATE handoffs SET state = 'accepted', accepted_by = ?1, accepted_at = ?2, \
         accepted_by_session = ?3 \
         WHERE id = ?4 AND state = 'open'",
        params![agent, now, session, handoff_id.as_bytes()],
    )?;
    Ok(())
}

fn observation_kind_as_str(kind: ObservationKind) -> &'static str {
    kind.as_str()
}

fn audit(
    tx: &rusqlite::Transaction<'_>,
    op: &str,
    workspace_id: Option<&[u8; 16]>,
    project_id: Option<&[u8; 16]>,
    page_id: Option<&[u8; 16]>,
    at: i64,
) -> StoreResult<()> {
    tx.execute(
        "INSERT INTO audit_log (at, op, workspace_id, project_id, page_id, detail) \
         VALUES (?1, ?2, ?3, ?4, ?5, '{}')",
        params![
            at,
            op,
            workspace_id.map(|b| &b[..]),
            project_id.map(|b| &b[..]),
            page_id.map(|b| &b[..])
        ],
    )?;
    Ok(())
}

/// Retro-fit sessions + observations to per-cwd projects and graveyard
/// any `is_latest=1` pages (which are mash-ups across the old single-project
/// bucket). Executes atomically in one transaction.
///
/// `plan` contains `(session_id, new_project_id)` pairs. Sessions not in
/// the plan are left untouched. Pages are graveyarded unconditionally so a
/// fresh consolidation can regenerate clean per-project pages.
pub fn reorg_sessions(
    conn: &mut Connection,
    plan: &[(SessionId, ProjectId)],
) -> StoreResult<ReorgSummary> {
    if plan.is_empty() {
        return Ok(ReorgSummary::default());
    }
    let tx = conn.transaction()?;
    let mut sessions_moved = 0usize;
    let mut observations_updated = 0usize;
    for (session_id, new_project_id) in plan {
        let rows = tx.execute(
            "UPDATE sessions SET project_id = ?1 WHERE id = ?2 AND project_id != ?1",
            params![new_project_id.as_bytes(), session_id.as_bytes()],
        )?;
        sessions_moved += rows;
        // Update observations whose session_id matches, keeping project_id
        // in sync with the session row we just moved.
        let obs_rows = tx.execute(
            "UPDATE observations SET project_id = ?1 WHERE session_id = ?2",
            params![new_project_id.as_bytes(), session_id.as_bytes()],
        )?;
        observations_updated += obs_rows;
    }
    // Graveyard all current latest pages — they mixed observations from
    // multiple projects and must be regenerated per-project by the next
    // consolidation pass.
    let pages_graveyarded: usize =
        tx.execute("UPDATE pages SET is_latest = 0 WHERE is_latest = 1", [])?;
    tx.commit()?;
    Ok(ReorgSummary {
        sessions_moved,
        observations_updated,
        pages_graveyarded,
    })
}

/// Rename a project within its workspace.
///
/// Only the `name` column is updated — all pages, sessions, observations,
/// and handoffs remain associated with the same `project_id`. No files
/// move on disk (the wiki is flat: every page from every project lives
/// under `wiki/`; only the `project_id` foreign key distinguishes them).
///
/// # Errors
/// - [`StoreError::InvalidProjectName`] when `new_name` is empty,
///   contains a `/` character, or is all whitespace.
/// - [`StoreError::ProjectNameTaken`] when a project with `new_name`
///   already exists in the same workspace.
/// - [`StoreError::Sqlite`] on any other SQL failure.
pub fn rename_project(
    conn: &mut Connection,
    workspace_id: &WorkspaceId,
    project_id: &ProjectId,
    new_name: &str,
) -> StoreResult<()> {
    let trimmed = new_name.trim();
    if trimmed.is_empty() {
        return Err(StoreError::InvalidProjectName(
            "project name must not be empty or all whitespace".into(),
        ));
    }
    if trimmed.contains('/') {
        return Err(StoreError::InvalidProjectName(
            "project name must not contain '/' (it appears in URL paths)".into(),
        ));
    }

    let rows = conn.execute(
        "UPDATE projects SET name = ?1 WHERE id = ?2 AND workspace_id = ?3",
        params![trimmed, project_id.as_bytes(), workspace_id.as_bytes()],
    );

    match rows {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(err, _))
            if err.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE
                || err.code == rusqlite::ErrorCode::ConstraintViolation =>
        {
            Err(StoreError::ProjectNameTaken(trimmed.to_string()))
        }
        Err(e) => Err(StoreError::Sqlite(e)),
    }
}

/// Record a successfully-applied wiki-structure migration.
///
/// Uses `INSERT OR IGNORE` so re-running the same name is a no-op
/// (idempotent by design — the runner already skips known names, but
/// this guards against any concurrent writes).
pub fn insert_wiki_migration(
    conn: &mut Connection,
    name: &str,
    applied_at: i64,
) -> StoreResult<()> {
    conn.execute(
        "INSERT OR IGNORE INTO wiki_migrations (name, applied_at) VALUES (?1, ?2)",
        params![name, applied_at],
    )?;
    Ok(())
}

/// Delete a project and all its data inside one transaction.
///
/// Execution order:
/// 1. Count rows in each dependent table (pages/all versions, sessions,
///    observations, handoffs, embeddings) before the delete so we can
///    report how many rows were removed.
/// 2. Collect all distinct page paths stored under the project — these are
///    the on-disk files the caller must clean up after this function returns.
/// 3. DELETE FROM projects WHERE id = ? — the ON DELETE CASCADE clauses in
///    V01 + V02 propagate the delete to pages, sessions, observations,
///    handoffs, and page_embeddings automatically.
/// 4. Commit and return the [`PurgeSummary`].
///
/// The `workspace_project_label` string is passed in by the caller (the
/// admin handler has the human-readable names; the writer only has IDs) and
/// forwarded verbatim into [`PurgeSummary::label`] for logging.
///
/// # Errors
/// Returns [`StoreError`] if any SQL statement fails. The transaction is
/// rolled back automatically on error.
pub fn purge_project(
    conn: &mut Connection,
    workspace_id: &WorkspaceId,
    project_id: &ProjectId,
    workspace_project_label: &str,
) -> StoreResult<PurgeSummary> {
    let tx = conn.transaction()?;

    let count = |sql: &str, id: &[u8]| -> StoreResult<u64> {
        let n: Option<i64> = tx
            .query_row(sql, rusqlite::params![id], |row| row.get(0))
            .optional()?;
        Ok(u64::try_from(n.unwrap_or(0)).unwrap_or(0))
    };

    let pid = project_id.as_bytes();
    let pages_deleted = count("SELECT COUNT(*) FROM pages WHERE project_id = ?1", &pid[..])?;
    let sessions_deleted = count(
        "SELECT COUNT(*) FROM sessions WHERE project_id = ?1",
        &pid[..],
    )?;
    let observations_deleted = count(
        "SELECT COUNT(*) FROM observations WHERE project_id = ?1",
        &pid[..],
    )?;
    let handoffs_deleted = count(
        "SELECT COUNT(*) FROM handoffs WHERE project_id = ?1",
        &pid[..],
    )?;
    // page_embeddings cascade through pages; count pages that have them.
    let embeddings_deleted = count(
        "SELECT COUNT(*) FROM page_embeddings \
         WHERE page_id IN (SELECT id FROM pages WHERE project_id = ?1)",
        &pid[..],
    )?;

    // Collect all distinct on-disk paths for the caller to clean up.
    // We use DISTINCT because multiple versions of the same logical page
    // share a path; the file only exists once. The statement must be
    // dropped before we call tx.commit() to release the borrow on `tx`.
    let page_paths: Vec<String> = {
        let mut path_stmt = tx.prepare("SELECT DISTINCT path FROM pages WHERE project_id = ?1")?;
        path_stmt
            .query_map(rusqlite::params![&pid[..]], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<String>>>()?
    };

    // Cascade handles pages / sessions / observations / handoffs /
    // page_embeddings. The workspace row is intentionally left intact —
    // other projects may still live there.
    tx.execute(
        "DELETE FROM projects WHERE id = ?1 AND workspace_id = ?2",
        rusqlite::params![&pid[..], workspace_id.as_bytes()],
    )?;

    tx.commit()?;
    Ok(PurgeSummary {
        label: workspace_project_label.to_string(),
        page_paths,
        pages_deleted,
        sessions_deleted,
        observations_deleted,
        handoffs_deleted,
        embeddings_deleted,
    })
}

#[cfg(test)]
mod tests {
    //! Focused unit tests for the load-bearing mutating SQL paths.
    //!
    //! `Store::open` exercises these incidentally through
    //! integration tests, but specific edges — supersession on body
    //! change, no-op on identical body, handoff state transitions,
    //! end_session summary linkage, embedding PK-replacement —
    //! deserve direct coverage so a regression surfaces with a
    //! one-line diff instead of a cascading e2e failure.
    use super::*;
    use ai_memory_core::{NewHandoff, NewPage, NewSession, PagePath, Tier};
    use rusqlite::Connection;
    use tempfile::TempDir;

    /// Open a fresh DB with migrations applied + a default workspace
    /// and "scratch" project pre-created. Tuple-return keeps the
    /// tempdir alive for the duration of the test.
    fn fresh_db() -> (
        TempDir,
        Connection,
        ai_memory_core::WorkspaceId,
        ai_memory_core::ProjectId,
    ) {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("test.sqlite");
        let mut conn = Connection::open(&db_path).unwrap();
        conn.pragma_update(None, "foreign_keys", "ON").unwrap();
        crate::migrations::run(&mut conn).unwrap();
        let ws = get_or_create_workspace(&mut conn, "default").unwrap();
        let proj = get_or_create_project(&mut conn, &ws, "scratch", None).unwrap();
        (tmp, conn, ws, proj)
    }

    fn page(
        ws: ai_memory_core::WorkspaceId,
        proj: ai_memory_core::ProjectId,
        path: &str,
        body: &str,
    ) -> NewPage {
        NewPage {
            workspace_id: ws,
            project_id: proj,
            path: PagePath::new(path).unwrap(),
            title: "test".into(),
            body: body.into(),
            tier: Tier::Semantic,
            frontmatter_json: serde_json::json!({}),
            pinned: false,
            links: Vec::new(),
        }
    }

    /// Trickier path: upserting a page with a CHANGED body must
    /// produce a NEW row and mark the previous row `is_latest = 0`.
    /// This is the M7 supersession chain — the entire wiki versioning
    /// guarantee rides on it.
    #[test]
    fn upsert_page_supersedes_on_body_change() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let id1 = upsert_page(&mut conn, &page(ws, proj, "notes/foo.md", "v1 body")).unwrap();
        let id2 = upsert_page(&mut conn, &page(ws, proj, "notes/foo.md", "v2 body")).unwrap();

        assert_ne!(id1, id2, "supersession must produce a new row id");

        let latest_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages WHERE path = ?1 AND is_latest = 1",
                params!["notes/foo.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(latest_count, 1, "exactly one latest version expected");

        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages WHERE path = ?1",
                params!["notes/foo.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(total, 2, "old version must remain on disk for history");

        // The newest row should point at the older as its predecessor
        // (supersedes column), so chains are reconstructible.
        let supersedes: Option<Vec<u8>> = conn
            .query_row(
                "SELECT supersedes FROM pages WHERE id = ?1",
                params![&id2.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert!(supersedes.is_some(), "new row must link to its predecessor");
    }

    /// Idempotency: re-upserting the same body should NOT create a
    /// second row. The watcher's reconciliation calls upsert_page on
    /// every file on every tick — without this, a quiet repo would
    /// accumulate spurious history every 30 seconds.
    #[test]
    fn upsert_page_is_noop_when_body_unchanged() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let p = page(ws, proj, "notes/foo.md", "same body");
        let id1 = upsert_page(&mut conn, &p).unwrap();
        let id2 = upsert_page(&mut conn, &p).unwrap();

        assert_eq!(id1, id2, "identical body should not supersede");
        conn.execute(
            "UPDATE pages SET updated_at = 123 WHERE id = ?1",
            params![id1.as_bytes()],
        )
        .unwrap();
        let id3 = upsert_page(&mut conn, &p).unwrap();
        assert_eq!(id1, id3, "identical body should keep the same page id");
        let updated_at: i64 = conn
            .query_row(
                "SELECT updated_at FROM pages WHERE id = ?1",
                params![id1.as_bytes()],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            updated_at, 123,
            "unchanged content should not dirty the row"
        );
        let total: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pages WHERE path = ?1",
                params!["notes/foo.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(total, 1, "no duplicate row for unchanged content");
    }

    #[test]
    fn upsert_page_persists_and_resolves_links() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let mut source = page(ws, proj, "concepts/source.md", "see target");
        source.links = vec![PagePath::new("decisions/target.md").unwrap()];
        let source_id = upsert_page(&mut conn, &source).unwrap();

        let unresolved: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM links \
                 WHERE from_page_id = ?1 AND to_path = ?2 AND to_page_id IS NULL",
                params![source_id.as_bytes(), "decisions/target.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(unresolved, 1, "forward link should persist unresolved");

        let target_id = upsert_page(
            &mut conn,
            &page(ws, proj, "decisions/target.md", "target body"),
        )
        .unwrap();

        let resolved: Option<Vec<u8>> = conn
            .query_row(
                "SELECT to_page_id FROM links WHERE from_page_id = ?1 AND to_path = ?2",
                params![source_id.as_bytes(), "decisions/target.md"],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(resolved.as_deref(), Some(&target_id.as_bytes()[..]));
    }

    /// Handoff state machine: insert → Open; accept_handoff → Accepted
    /// with accepted_by stamped. Calling accept again must be safe
    /// (idempotent at the DB level) because hooks fire-and-forget.
    #[test]
    fn accept_handoff_transitions_open_to_accepted() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let new = NewHandoff {
            workspace_id: ws,
            project_id: proj,
            from_session_id: None,
            from_agent: AgentKind::ClaudeCode,
            to_agent: None,
            cwd: None,
            summary: "test summary".into(),
            open_questions: vec![],
            next_steps: vec![],
            files_touched: vec![],
        };
        let id = insert_handoff(&mut conn, &new).unwrap();

        // Pre-state: Open, accepted_by NULL.
        let (state, accepted_by): (String, Option<String>) = conn
            .query_row(
                "SELECT state, accepted_by FROM handoffs WHERE id = ?1",
                params![&id.as_bytes()[..]],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(state, "open");
        assert!(accepted_by.is_none());

        accept_handoff(&mut conn, &id, AgentKind::Codex, None).unwrap();
        let (state, accepted_by): (String, Option<String>) = conn
            .query_row(
                "SELECT state, accepted_by FROM handoffs WHERE id = ?1",
                params![&id.as_bytes()[..]],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(state, "accepted");
        assert_eq!(accepted_by.as_deref(), Some("codex"));

        // Idempotency: accepting an already-accepted handoff must
        // either succeed silently or fail clearly, never corrupt
        // the row. (Current impl is a no-op UPDATE with a state
        // guard.)
        let second = accept_handoff(&mut conn, &id, AgentKind::Codex, None);
        assert!(second.is_ok(), "double-accept must not error");
    }

    /// Pi / OMP hooks persist `agent_kind = 'omp'`. V01's CHECK omitted
    /// that value; regression for the hook-router WARN on every event.
    #[test]
    fn begin_session_accepts_omp_agent_kind() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let sid = SessionId::new();
        begin_session(
            &mut conn,
            &NewSession {
                id: sid,
                workspace_id: ws,
                project_id: proj,
                agent_kind: AgentKind::Omp,
                cwd: Some(std::path::PathBuf::from(r"C:\GIT\ai-memory")),
            },
        )
        .unwrap();

        let stored: String = conn
            .query_row(
                "SELECT agent_kind FROM sessions WHERE id = ?1",
                params![&sid.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(stored, "omp");
    }

    /// end_session links the synthesised summary page so callers can
    /// jump straight from session row to summary.
    #[test]
    fn end_session_links_summary_page_when_provided() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let sid = SessionId::new();
        begin_session(
            &mut conn,
            &NewSession {
                id: sid,
                workspace_id: ws,
                project_id: proj,
                agent_kind: AgentKind::ClaudeCode,
                cwd: None,
            },
        )
        .unwrap();
        let page_id = upsert_page(
            &mut conn,
            &page(ws, proj, "sessions/abc.md", "summary body"),
        )
        .unwrap();
        end_session(&mut conn, &sid, Some(&page_id)).unwrap();

        let summary: Option<Vec<u8>> = conn
            .query_row(
                "SELECT summary_page_id FROM sessions WHERE id = ?1",
                params![&sid.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert!(
            summary.is_some(),
            "summary_page_id must persist when supplied"
        );
        let bytes = summary.unwrap();
        assert_eq!(bytes.len(), 16);
        assert_eq!(&bytes[..], &page_id.as_bytes()[..]);
    }

    /// end_session without a summary leaves the column NULL — the
    /// session ended but no page was synthesised (e.g. zero
    /// observations recorded). This must not be confused with the
    /// summary-linked case.
    #[test]
    fn end_session_without_summary_page_id_leaves_null() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let sid = SessionId::new();
        begin_session(
            &mut conn,
            &NewSession {
                id: sid,
                workspace_id: ws,
                project_id: proj,
                agent_kind: AgentKind::ClaudeCode,
                cwd: None,
            },
        )
        .unwrap();
        end_session(&mut conn, &sid, None).unwrap();
        let summary: Option<Vec<u8>> = conn
            .query_row(
                "SELECT summary_page_id FROM sessions WHERE id = ?1",
                params![&sid.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert!(summary.is_none());
    }

    /// Embeddings are keyed by page_id (PK). Re-storing for the same
    /// page must REPLACE, not duplicate — otherwise `ai-memory embed
    /// --reembed` would multiply rows on each run.
    #[test]
    fn store_embedding_replaces_existing_row() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let pid = upsert_page(&mut conn, &page(ws, proj, "notes/x.md", "body")).unwrap();
        store_embedding(
            &mut conn,
            &pid,
            &vec![0u8; 1536 * 4],
            "test",
            "model-a",
            1536,
        )
        .unwrap();
        store_embedding(
            &mut conn,
            &pid,
            &vec![1u8; 1536 * 4],
            "test",
            "model-b",
            1536,
        )
        .unwrap();

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM page_embeddings WHERE page_id = ?1",
                params![&pid.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1, "embedding row must be replaced, not duplicated");

        let model: String = conn
            .query_row(
                "SELECT model FROM page_embeddings WHERE page_id = ?1",
                params![&pid.as_bytes()[..]],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(model, "model-b", "latest model metadata wins");
    }

    #[test]
    fn store_embeddings_batches_rows_in_one_call() {
        let (_tmp, mut conn, ws, proj) = fresh_db();
        let p1 = upsert_page(&mut conn, &page(ws, proj, "notes/a.md", "body a")).unwrap();
        let p2 = upsert_page(&mut conn, &page(ws, proj, "notes/b.md", "body b")).unwrap();

        store_embeddings(
            &mut conn,
            &[
                EmbeddingWrite {
                    page_id: p1,
                    vector_bytes: vec![0u8; 4],
                    provider: "test".into(),
                    model: "model".into(),
                    dim: 1,
                },
                EmbeddingWrite {
                    page_id: p2,
                    vector_bytes: vec![1u8; 4],
                    provider: "test".into(),
                    model: "model".into(),
                    dim: 1,
                },
            ],
        )
        .unwrap();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM page_embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 2);
    }
}
