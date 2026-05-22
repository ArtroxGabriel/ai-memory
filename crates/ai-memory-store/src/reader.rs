//! Read-only connection pool and query helpers.
//!
//! WAL mode lets us have unlimited concurrent readers alongside the single
//! writer, so the pool is mostly about bounding file-descriptor usage and
//! avoiding `Connection::open` overhead on hot paths. Pool eviction is a
//! soft cap: a connection that comes back when the pool is already full
//! is simply dropped.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai_memory_core::{
    AgentKind, Handoff, HandoffId, HandoffState, Observation, ObservationId, ObservationKind,
    PageId, PagePath, ProjectId, SessionId, WorkspaceId,
};
use parking_lot::Mutex;
use rusqlite::{Connection, OpenFlags, OptionalExtension, params};
use serde::Serialize;
// `ai_memory_core::Tier` is referenced via fully-qualified path inside the
// DecayCandidate struct definition above to avoid a top-level import
// for a single use-site.

use crate::error::{StoreError, StoreResult};

/// One hit returned by [`ReaderPool::search_pages`].
#[derive(Debug, Clone, Serialize)]
pub struct PageHit {
    /// Stable identifier for this page version.
    pub id: PageId,
    /// Relative path within the wiki tree.
    pub path: PagePath,
    /// Page title.
    pub title: String,
    /// FTS5 snippet of the body around the matched terms (HTML-marked).
    pub snippet: String,
    /// FTS5 rank score (lower is better — closer to query terms).
    pub rank: f64,
}

/// Aggregate counts surfaced by [`ReaderPool::status_counts`] and consumed
/// by `ai-memory status`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StatusCounts {
    /// Pages with `is_latest = 1`.
    pub pages_latest: u64,
    /// All page versions including superseded ones.
    pub pages_all: u64,
    /// Total sessions ever recorded.
    pub sessions: u64,
    /// Total observations across all sessions.
    pub observations: u64,
}

/// Cheap, cloneable read-only connection pool handle.
#[derive(Clone)]
pub struct ReaderPool {
    inner: Arc<Inner>,
}

struct Inner {
    db_path: PathBuf,
    pool: Mutex<Vec<Connection>>,
    soft_cap: usize,
}

impl ReaderPool {
    /// Initialise the pool. Connections are opened lazily on first use.
    ///
    /// # Errors
    /// Currently infallible, but reserved so we can pre-open connections
    /// in a later milestone.
    pub fn new(db_path: &Path, soft_cap: usize) -> StoreResult<Self> {
        Ok(Self {
            inner: Arc::new(Inner {
                db_path: db_path.to_path_buf(),
                pool: Mutex::new(Vec::with_capacity(soft_cap.max(1))),
                soft_cap: soft_cap.max(1),
            }),
        })
    }

    /// Run a synchronous closure against a pooled read-only connection.
    ///
    /// The closure runs on the tokio blocking pool so it never starves the
    /// async runtime. If the pool is empty we open a fresh connection;
    /// on return we keep it only when the pool is below its soft cap.
    ///
    /// # Errors
    /// Returns [`StoreError::PoolPanic`] if the blocking task panics; any
    /// error returned by the closure is propagated unchanged.
    pub async fn with_conn<F, T>(&self, f: F) -> StoreResult<T>
    where
        F: FnOnce(&Connection) -> StoreResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let conn = checkout(&inner)?;
            let result = f(&conn);
            checkin(&inner, conn);
            result
        })
        .await
        .map_err(|e| StoreError::PoolPanic(e.to_string()))?
    }

    /// Run a full-text search against the FTS5 index and return the top
    /// matches, limited to `is_latest = 1` rows.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn search_pages(&self, query: String, limit: usize) -> StoreResult<Vec<PageHit>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT pages.id, pages.path, pages.title, \
                        snippet(pages_fts, 1, '<mark>', '</mark>', '…', 24) AS snip, \
                        pages_fts.rank \
                 FROM pages_fts \
                 JOIN pages ON pages.rowid = pages_fts.rowid \
                 WHERE pages_fts MATCH ?1 AND pages.is_latest = 1 \
                 ORDER BY pages_fts.rank \
                 LIMIT ?2",
            )?;
            #[allow(clippy::cast_possible_wrap)]
            let rows = stmt.query_map(params![query, limit as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let path: String = row.get(1)?;
                let title: String = row.get(2)?;
                let snippet: String = row.get(3)?;
                let rank: f64 = row.get(4)?;
                Ok((id_bytes, path, title, snippet, rank))
            })?;

            let mut hits = Vec::new();
            for row in rows {
                let (id_bytes, path, title, snippet, rank) = row?;
                hits.push(PageHit {
                    id: PageId::from_slice(&id_bytes)?,
                    path: PagePath::new(path)?,
                    title,
                    snippet,
                    rank,
                });
            }
            Ok(hits)
        })
        .await
    }

    /// Return the N most-recently-updated `is_latest = 1` pages.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn recent_pages(&self, limit: usize) -> StoreResult<Vec<PageHit>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, path, title, \
                        substr(body, 1, 240) AS snip, \
                        CAST(updated_at AS REAL) AS rank \
                 FROM pages \
                 WHERE is_latest = 1 \
                 ORDER BY updated_at DESC \
                 LIMIT ?1",
            )?;
            #[allow(clippy::cast_possible_wrap)]
            let rows = stmt.query_map(params![limit as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let path: String = row.get(1)?;
                let title: String = row.get(2)?;
                let snippet: String = row.get(3)?;
                let rank: f64 = row.get(4)?;
                Ok((id_bytes, path, title, snippet, rank))
            })?;
            let mut hits = Vec::new();
            for row in rows {
                let (id_bytes, path, title, snippet, rank) = row?;
                hits.push(PageHit {
                    id: PageId::from_slice(&id_bytes)?,
                    path: PagePath::new(path)?,
                    title,
                    snippet,
                    rank,
                });
            }
            Ok(hits)
        })
        .await
    }

    /// Return all observations for the given session, ordered by
    /// `created_at` ascending.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn observations_for_session(
        &self,
        session_id: SessionId,
    ) -> StoreResult<Vec<Observation>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, session_id, workspace_id, project_id, kind, title, body, \
                        importance, created_at \
                 FROM observations \
                 WHERE session_id = ?1 \
                 ORDER BY created_at ASC",
            )?;
            let rows = stmt.query_map(params![session_id.as_bytes()], row_to_observation)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r??);
            }
            Ok(out)
        })
        .await
    }

    /// Load every `is_latest=1` page's embedding for the project, but
    /// only when the stored `(provider, model, dim)` matches the
    /// caller's expectation. Mismatched rows are skipped (the
    /// refuse-on-mismatch check is `embedding_meta_for_mismatch`).
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn load_embeddings(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        provider: String,
        model: String,
        dim: u32,
    ) -> StoreResult<Vec<StoredEmbedding>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT page_embeddings.page_id, page_embeddings.vector, pages.path \
                 FROM page_embeddings \
                 JOIN pages ON pages.id = page_embeddings.page_id \
                 WHERE pages.workspace_id = ?1 \
                   AND pages.project_id = ?2 \
                   AND pages.is_latest = 1 \
                   AND page_embeddings.provider = ?3 \
                   AND page_embeddings.model = ?4 \
                   AND page_embeddings.dim = ?5",
            )?;
            let rows = stmt.query_map(
                params![
                    workspace_id.as_bytes(),
                    project_id.as_bytes(),
                    provider,
                    model,
                    dim,
                ],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let vec_bytes: Vec<u8> = row.get(1)?;
                    let path: String = row.get(2)?;
                    Ok((id_bytes, vec_bytes, path))
                },
            )?;
            let mut out = Vec::new();
            for r in rows {
                let (id_bytes, vec_bytes, path) = r?;
                let id = PageId::from_slice(&id_bytes)?;
                let path = PagePath::new(path)?;
                let vector = bytes_to_f32_vec(&vec_bytes, dim)?;
                out.push(StoredEmbedding { id, path, vector });
            }
            Ok(out)
        })
        .await
    }

    /// Return any `(provider, model, dim)` triples currently stored
    /// that *don't* match the caller's expectation. An empty vec
    /// means "all clean". Used at startup for the refuse-on-mismatch
    /// (agentmemory #469 lesson).
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn embedding_meta_for_mismatch(
        &self,
        provider: String,
        model: String,
        dim: u32,
    ) -> StoreResult<Vec<(String, String, u32, u64)>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT provider, model, dim, COUNT(*) \
                 FROM page_embeddings \
                 WHERE NOT (provider = ?1 AND model = ?2 AND dim = ?3) \
                 GROUP BY provider, model, dim",
            )?;
            let rows = stmt.query_map(params![provider, model, dim], |row| {
                let provider: String = row.get(0)?;
                let model: String = row.get(1)?;
                let dim: i64 = row.get(2)?;
                let count: i64 = row.get(3)?;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                Ok((
                    provider,
                    model,
                    dim as u32,
                    u64::try_from(count).unwrap_or(0),
                ))
            })?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        })
        .await
    }

    /// Return decay-evaluation candidates for the M8 forget sweep.
    ///
    /// Walks `pages` rows with `is_latest = 1` and returns the columns
    /// the forget sweep needs to compute the retention formula. The
    /// sweep itself filters by tier (only `episodic`) + pinned flag,
    /// so this method does not pre-filter -- it just hands the data
    /// over.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn decay_candidates(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
    ) -> StoreResult<Vec<DecayCandidate>> {
        self.with_conn(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, path, tier, pinned, updated_at, access_count, last_accessed_at, \
                        frontmatter_json \
                 FROM pages \
                 WHERE workspace_id = ?1 AND project_id = ?2 AND is_latest = 1",
            )?;
            let rows = stmt.query_map(
                params![workspace_id.as_bytes(), project_id.as_bytes()],
                row_to_decay_candidate,
            )?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r??);
            }
            Ok(out)
        })
        .await
    }

    /// Hybrid search: RRF-fuse FTS5 results with cosine-similarity
    /// over the stored embeddings of the matching `(provider, model,
    /// dim)`. Returns the top-`limit` pages by fused score.
    ///
    /// When `query_vec` is `None`, falls back to pure FTS5
    /// (equivalent to [`Self::search_pages`]). When no embeddings
    /// exist for the configured triple, also falls back to pure FTS5.
    ///
    /// k=60 is the canonical RRF constant.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    #[allow(clippy::too_many_arguments)]
    pub async fn hybrid_search(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        query: String,
        query_vec: Option<Vec<f32>>,
        provider: String,
        model: String,
        dim: u32,
        limit: usize,
    ) -> StoreResult<Vec<PageHit>> {
        // Fetch FTS5 hits first.
        let fts_hits = self.search_pages(query, limit * 2).await?;
        let Some(qv) = query_vec else {
            // No query vector → pure FTS5.
            let mut out = fts_hits;
            out.truncate(limit);
            return Ok(out);
        };
        // Compute cosines against all stored embeddings.
        let stored = self
            .load_embeddings(workspace_id, project_id, provider, model, dim)
            .await?;
        if stored.is_empty() {
            let mut out = fts_hits;
            out.truncate(limit);
            return Ok(out);
        }
        let mut vec_hits: Vec<(PageId, PagePath, f32)> = stored
            .iter()
            .map(|s| {
                let score = dot(&qv, &s.vector);
                (s.id, s.path.clone(), score)
            })
            .collect();
        vec_hits.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        vec_hits.truncate(limit * 2);

        // RRF fuse: score(d) = Σ 1/(k + rank_i(d)) over rankers.
        let k = 60.0_f64;
        let mut fused: std::collections::HashMap<PageId, (PagePath, String, String, f64, f64)> =
            std::collections::HashMap::new();

        for (rank, h) in fts_hits.iter().enumerate() {
            let contrib = 1.0 / (k + (rank + 1) as f64);
            fused
                .entry(h.id)
                .and_modify(|entry| entry.3 += contrib)
                .or_insert((
                    h.path.clone(),
                    h.title.clone(),
                    h.snippet.clone(),
                    contrib,
                    h.rank,
                ));
        }
        for (rank, (id, path, _score)) in vec_hits.iter().enumerate() {
            let contrib = 1.0 / (k + (rank + 1) as f64);
            fused
                .entry(*id)
                .and_modify(|entry| entry.3 += contrib)
                .or_insert((path.clone(), String::new(), String::new(), contrib, 0.0));
        }

        let mut out: Vec<PageHit> = fused
            .into_iter()
            .map(|(id, (path, title, snippet, fused_rank, _orig))| PageHit {
                id,
                path,
                title,
                snippet,
                rank: -fused_rank, // lower = better (matches FTS5 convention)
            })
            .collect();
        out.sort_by(|a, b| {
            a.rank
                .partial_cmp(&b.rank)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(limit);
        Ok(out)
    }

    /// Return the latest open handoff for the project, optionally
    /// filtered to a specific `cwd`.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn latest_open_handoff(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        cwd_filter: Option<String>,
    ) -> StoreResult<Option<Handoff>> {
        self.with_conn(move |conn| {
            let mut stmt: rusqlite::Statement<'_> = if let Some(_cwd) = cwd_filter.as_deref() {
                conn.prepare(
                    "SELECT id, workspace_id, project_id, from_session_id, from_agent, to_agent, \
                            cwd, summary, open_questions, next_steps, files_touched, state, \
                            created_at, accepted_by, accepted_at, accepted_by_session \
                     FROM handoffs \
                     WHERE workspace_id = ?1 AND project_id = ?2 AND cwd = ?3 \
                       AND state = 'open' \
                     ORDER BY created_at DESC LIMIT 1",
                )?
            } else {
                conn.prepare(
                    "SELECT id, workspace_id, project_id, from_session_id, from_agent, to_agent, \
                            cwd, summary, open_questions, next_steps, files_touched, state, \
                            created_at, accepted_by, accepted_at, accepted_by_session \
                     FROM handoffs \
                     WHERE workspace_id = ?1 AND project_id = ?2 AND state = 'open' \
                     ORDER BY created_at DESC LIMIT 1",
                )?
            };
            let row_opt = if let Some(c) = cwd_filter.as_deref() {
                stmt.query_row(
                    params![workspace_id.as_bytes(), project_id.as_bytes(), c],
                    row_to_handoff,
                )
                .optional()?
            } else {
                stmt.query_row(
                    params![workspace_id.as_bytes(), project_id.as_bytes()],
                    row_to_handoff,
                )
                .optional()?
            };
            row_opt.transpose()
        })
        .await
    }

    /// Look up a handoff by id.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn handoff_by_id(&self, handoff_id: HandoffId) -> StoreResult<Option<Handoff>> {
        self.with_conn(move |conn| {
            let row = conn
                .query_row(
                    "SELECT id, workspace_id, project_id, from_session_id, from_agent, to_agent, \
                            cwd, summary, open_questions, next_steps, files_touched, state, \
                            created_at, accepted_by, accepted_at, accepted_by_session \
                     FROM handoffs WHERE id = ?1",
                    params![handoff_id.as_bytes()],
                    row_to_handoff,
                )
                .optional()?;
            row.transpose()
        })
        .await
    }

    /// Snapshot the database to `dest_path` using SQLite's online backup
    /// API. The source DB stays writable for the duration of the copy.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn snapshot_to(&self, dest_path: PathBuf) -> StoreResult<()> {
        self.with_conn(move |conn| {
            conn.backup(rusqlite::DatabaseName::Main, &dest_path, None)
                .map_err(StoreError::from)
        })
        .await
    }

    /// Return aggregate counts for the `status` view.
    ///
    /// # Errors
    /// Propagates any SQL or pool error.
    pub async fn status_counts(&self) -> StoreResult<StatusCounts> {
        self.with_conn(|conn| {
            let pages_latest: u64 = count(conn, "SELECT COUNT(*) FROM pages WHERE is_latest = 1")?;
            let pages_all: u64 = count(conn, "SELECT COUNT(*) FROM pages")?;
            let sessions: u64 = count(conn, "SELECT COUNT(*) FROM sessions")?;
            let observations: u64 = count(conn, "SELECT COUNT(*) FROM observations")?;
            Ok(StatusCounts {
                pages_latest,
                pages_all,
                sessions,
                observations,
            })
        })
        .await
    }
}

fn row_to_observation(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoreResult<Observation>> {
    let id_bytes: Vec<u8> = row.get(0)?;
    let session_bytes: Vec<u8> = row.get(1)?;
    let workspace_bytes: Vec<u8> = row.get(2)?;
    let project_bytes: Vec<u8> = row.get(3)?;
    let kind_str: String = row.get(4)?;
    let title: String = row.get(5)?;
    let body: String = row.get(6)?;
    let importance: i64 = row.get(7)?;
    let created_us: i64 = row.get(8)?;
    Ok(materialise_observation(
        id_bytes,
        session_bytes,
        workspace_bytes,
        project_bytes,
        kind_str,
        title,
        body,
        importance,
        created_us,
    ))
}

#[allow(clippy::too_many_arguments)]
fn materialise_observation(
    id_bytes: Vec<u8>,
    session_bytes: Vec<u8>,
    workspace_bytes: Vec<u8>,
    project_bytes: Vec<u8>,
    kind_str: String,
    title: String,
    body: String,
    importance: i64,
    created_us: i64,
) -> StoreResult<Observation> {
    Ok(Observation {
        id: ObservationId::from_slice(&id_bytes)?,
        session_id: SessionId::from_slice(&session_bytes)?,
        workspace_id: WorkspaceId::from_slice(&workspace_bytes)?,
        project_id: ProjectId::from_slice(&project_bytes)?,
        kind: kind_str
            .parse::<ObservationKind>()
            .map_err(StoreError::from)?,
        title,
        body,
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        importance: importance.clamp(1, 10) as u8,
        created_at: jiff::Timestamp::from_microsecond(created_us).map_err(|e| {
            StoreError::Memory(ai_memory_core::MemoryError::MalformedRecord(format!(
                "bad timestamp: {e}"
            )))
        })?,
    })
}

/// One stored embedding row, materialised for the vector path.
#[derive(Debug, Clone)]
pub struct StoredEmbedding {
    /// Page identifier (always the `is_latest=1` row's id).
    pub id: PageId,
    /// Relative wiki path.
    pub path: PagePath,
    /// Unit-normalised vector.
    pub vector: Vec<f32>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn bytes_to_f32_vec(bytes: &[u8], dim: u32) -> StoreResult<Vec<f32>> {
    let expected = (dim as usize) * 4;
    if bytes.len() != expected {
        return Err(StoreError::Memory(
            ai_memory_core::MemoryError::MalformedRecord(format!(
                "embedding bytes {} != expected {}",
                bytes.len(),
                expected
            )),
        ));
    }
    let mut out = Vec::with_capacity(dim as usize);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

/// Pack a `&[f32]` into little-endian bytes for storage. Inverse of
/// [`bytes_to_f32_vec`].
#[must_use]
pub fn f32_vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// One row's worth of input for the M8 retention formula.
#[derive(Debug, Clone, Serialize)]
pub struct DecayCandidate {
    /// Stable identifier.
    pub id: PageId,
    /// Relative wiki path.
    pub path: PagePath,
    /// Tier (the sweep only considers `episodic`).
    pub tier: ai_memory_core::Tier,
    /// Pinned flag — true means "never decay".
    pub pinned: bool,
    /// `updated_at` in microseconds since epoch.
    pub updated_at_us: i64,
    /// Total query/access hits.
    pub access_count: u32,
    /// `last_accessed_at` in microseconds since epoch, or `None` if never accessed.
    pub last_accessed_at_us: Option<i64>,
    /// Frontmatter JSON; the sweep peeks at it for an explicit
    /// `pinned: true` (which overrides the schema flag).
    pub frontmatter_json: String,
}

fn row_to_decay_candidate(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<StoreResult<DecayCandidate>> {
    let id_bytes: Vec<u8> = row.get(0)?;
    let path: String = row.get(1)?;
    let tier_str: String = row.get(2)?;
    let pinned: i64 = row.get(3)?;
    let updated_at_us: i64 = row.get(4)?;
    let access_count: i64 = row.get(5)?;
    let last_accessed_at_us: Option<i64> = row.get(6)?;
    let frontmatter_json: String = row.get(7)?;
    Ok(materialise_decay_candidate(
        id_bytes,
        path,
        tier_str,
        pinned,
        updated_at_us,
        access_count,
        last_accessed_at_us,
        frontmatter_json,
    ))
}

#[allow(clippy::too_many_arguments)]
fn materialise_decay_candidate(
    id_bytes: Vec<u8>,
    path: String,
    tier_str: String,
    pinned: i64,
    updated_at_us: i64,
    access_count: i64,
    last_accessed_at_us: Option<i64>,
    frontmatter_json: String,
) -> StoreResult<DecayCandidate> {
    Ok(DecayCandidate {
        id: PageId::from_slice(&id_bytes)?,
        path: PagePath::new(path)?,
        tier: tier_str
            .parse::<ai_memory_core::Tier>()
            .map_err(StoreError::from)?,
        pinned: pinned != 0,
        updated_at_us,
        access_count: u32::try_from(access_count.max(0)).unwrap_or(u32::MAX),
        last_accessed_at_us,
        frontmatter_json,
    })
}

fn row_to_handoff(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoreResult<Handoff>> {
    let id_bytes: Vec<u8> = row.get(0)?;
    let ws_bytes: Vec<u8> = row.get(1)?;
    let pj_bytes: Vec<u8> = row.get(2)?;
    let from_session_bytes: Option<Vec<u8>> = row.get(3)?;
    let from_agent: String = row.get(4)?;
    let to_agent: Option<String> = row.get(5)?;
    let cwd: Option<String> = row.get(6)?;
    let summary: String = row.get(7)?;
    let open_q_json: String = row.get(8)?;
    let next_s_json: String = row.get(9)?;
    let files_json: String = row.get(10)?;
    let state: String = row.get(11)?;
    let created_us: i64 = row.get(12)?;
    let accepted_by: Option<String> = row.get(13)?;
    let accepted_at_us: Option<i64> = row.get(14)?;
    let accepted_by_session_bytes: Option<Vec<u8>> = row.get(15)?;
    Ok(materialise_handoff(
        id_bytes,
        ws_bytes,
        pj_bytes,
        from_session_bytes,
        from_agent,
        to_agent,
        cwd,
        summary,
        open_q_json,
        next_s_json,
        files_json,
        state,
        created_us,
        accepted_by,
        accepted_at_us,
        accepted_by_session_bytes,
    ))
}

#[allow(clippy::too_many_arguments)]
fn materialise_handoff(
    id_bytes: Vec<u8>,
    ws_bytes: Vec<u8>,
    pj_bytes: Vec<u8>,
    from_session_bytes: Option<Vec<u8>>,
    from_agent: String,
    to_agent: Option<String>,
    cwd: Option<String>,
    summary: String,
    open_q_json: String,
    next_s_json: String,
    files_json: String,
    state: String,
    created_us: i64,
    accepted_by: Option<String>,
    accepted_at_us: Option<i64>,
    accepted_by_session_bytes: Option<Vec<u8>>,
) -> StoreResult<Handoff> {
    let open_questions: Vec<String> = serde_json::from_str(&open_q_json)?;
    let next_steps: Vec<String> = serde_json::from_str(&next_s_json)?;
    let files_touched: Vec<String> = serde_json::from_str(&files_json)?;
    let from_session = from_session_bytes
        .as_deref()
        .map(SessionId::from_slice)
        .transpose()?;
    let accepted_session = accepted_by_session_bytes
        .as_deref()
        .map(SessionId::from_slice)
        .transpose()?;
    Ok(Handoff {
        id: HandoffId::from_slice(&id_bytes)?,
        workspace_id: WorkspaceId::from_slice(&ws_bytes)?,
        project_id: ProjectId::from_slice(&pj_bytes)?,
        from_session_id: from_session,
        from_agent: parse_agent(&from_agent),
        to_agent: to_agent.as_deref().map(parse_agent),
        cwd,
        summary,
        open_questions,
        next_steps,
        files_touched,
        state: state.parse::<HandoffState>().map_err(StoreError::from)?,
        created_at: jiff::Timestamp::from_microsecond(created_us).map_err(|e| {
            StoreError::Memory(ai_memory_core::MemoryError::MalformedRecord(format!(
                "bad created_at: {e}"
            )))
        })?,
        accepted_by: accepted_by.as_deref().map(parse_agent),
        accepted_at: accepted_at_us
            .map(jiff::Timestamp::from_microsecond)
            .transpose()
            .map_err(|e| {
                StoreError::Memory(ai_memory_core::MemoryError::MalformedRecord(format!(
                    "bad accepted_at: {e}"
                )))
            })?,
        accepted_by_session: accepted_session,
    })
}

fn parse_agent(s: &str) -> AgentKind {
    match s {
        "claude-code" => AgentKind::ClaudeCode,
        "codex" => AgentKind::Codex,
        "open-code" => AgentKind::OpenCode,
        _ => AgentKind::Other,
    }
}

fn count(conn: &Connection, sql: &str) -> StoreResult<u64> {
    let n: Option<i64> = conn.query_row(sql, [], |row| row.get(0)).optional()?;
    Ok(u64::try_from(n.unwrap_or(0)).unwrap_or(0))
}

fn checkout(inner: &Inner) -> StoreResult<Connection> {
    if let Some(conn) = inner.pool.lock().pop() {
        return Ok(conn);
    }
    open_read_only(&inner.db_path)
}

fn checkin(inner: &Inner, conn: Connection) {
    let mut pool = inner.pool.lock();
    if pool.len() < inner.soft_cap {
        pool.push(conn);
    }
}

fn open_read_only(path: &Path) -> StoreResult<Connection> {
    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY
        | OpenFlags::SQLITE_OPEN_URI
        | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let conn = Connection::open_with_flags(path, flags)?;
    conn.pragma_update(None, "busy_timeout", 5_000)?;
    Ok(conn)
}
