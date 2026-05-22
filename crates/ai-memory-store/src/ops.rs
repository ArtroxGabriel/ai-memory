//! Mutating SQL operations executed on the writer thread.
//!
//! Each operation is one transaction. Calling them from anywhere other than
//! the writer thread would violate the single-writer invariant (see
//! [`crate::writer`]).

use ai_memory_core::{
    AgentKind, HandoffId, NewHandoff, NewObservation, NewPage, NewSession, ObservationId,
    ObservationKind, PageId, SessionId,
};
use jiff::Timestamp;
use rusqlite::{Connection, OptionalExtension, params};
use sha2::{Digest, Sha256};

use crate::error::{StoreError, StoreResult};

/// Upsert a page by path, superseding any existing latest version when the
/// content (sha256 of body) has changed.
///
/// Returns the id of the page row that should now be considered current.
pub fn upsert_page(conn: &mut Connection, page: &NewPage) -> StoreResult<PageId> {
    let body_sha256: [u8; 32] = {
        let mut hasher = Sha256::new();
        hasher.update(page.body.as_bytes());
        hasher.finalize().into()
    };
    let frontmatter_str = serde_json::to_string(&page.frontmatter_json)?;
    let now = Timestamp::now().as_microsecond();
    let tier_str = page.tier.as_str();

    let tx = conn.transaction()?;

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

    let result_id = if let Some((existing_id, existing_sha)) = existing {
        if existing_sha == body_sha256 {
            // Content unchanged; touch updated_at only and return existing id.
            tx.execute(
                "UPDATE pages SET updated_at = ?1 WHERE id = ?2",
                params![now, existing_id],
            )?;
            PageId::from_slice(&existing_id)?
        } else {
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
            audit(
                &tx,
                "supersede_page",
                Some(page.workspace_id.as_bytes()),
                Some(page.project_id.as_bytes()),
                Some(new_id.as_bytes()),
                now,
            )?;
            new_id
        }
    } else {
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
        audit(
            &tx,
            "create_page",
            Some(page.workspace_id.as_bytes()),
            Some(page.project_id.as_bytes()),
            Some(new_id.as_bytes()),
            now,
        )?;
        new_id
    };

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
            tx.execute(
                "UPDATE pages SET updated_at = ?1 WHERE id = ?2",
                params![now, existing_id],
            )?;
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

/// Begin (or re-affirm) a session row keyed on the caller-supplied id.
/// Idempotent: a second call with the same id leaves the row untouched.
pub fn begin_session(conn: &mut Connection, session: &NewSession) -> StoreResult<()> {
    let now = Timestamp::now().as_microsecond();
    let agent = agent_kind_as_str(session.agent_kind);
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
    let from_agent = agent_kind_as_str(h.from_agent);
    let to_agent = h.to_agent.map(agent_kind_as_str);
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
    let agent = agent_kind_as_str(accepting_agent);
    let session: Option<&[u8]> = accepting_session.map(|s| &s.as_bytes()[..]);
    conn.execute(
        "UPDATE handoffs SET state = 'accepted', accepted_by = ?1, accepted_at = ?2, \
         accepted_by_session = ?3 \
         WHERE id = ?4 AND state = 'open'",
        params![agent, now, session, handoff_id.as_bytes()],
    )?;
    Ok(())
}

fn agent_kind_as_str(kind: AgentKind) -> &'static str {
    match kind {
        AgentKind::ClaudeCode => "claude-code",
        AgentKind::Codex => "codex",
        AgentKind::OpenCode => "open-code",
        AgentKind::Other => "other",
    }
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
