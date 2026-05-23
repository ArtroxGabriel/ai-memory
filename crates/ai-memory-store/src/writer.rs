//! Single-writer SQLite actor.
//!
//! Every mutating SQL statement flows through one dedicated OS thread that
//! owns the writer [`rusqlite::Connection`]. Callers send [`WriteCmd`]
//! variants over an mpsc channel and receive results back through a
//! `oneshot`. This pattern eliminates the `database is locked` failure
//! mode that bit cognee (#2717) — there is exactly one writer at all
//! times, by construction.

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use ai_memory_core::{
    AgentKind, HandoffId, NewHandoff, NewObservation, NewPage, NewSession, ObservationId, PageId,
    ProjectId, SessionId, WorkspaceId,
};
use rusqlite::Connection;
use tokio::sync::{mpsc, oneshot};

use crate::error::{StoreError, StoreResult};
use crate::ops::{self, ReorgSummary};

/// Commands accepted by the writer thread.
pub(crate) enum WriteCmd {
    GetOrCreateWorkspace {
        name: String,
        reply: oneshot::Sender<StoreResult<WorkspaceId>>,
    },
    GetOrCreateProject {
        workspace_id: WorkspaceId,
        name: String,
        repo_path: Option<String>,
        reply: oneshot::Sender<StoreResult<ProjectId>>,
    },
    UpsertPage {
        page: NewPage,
        reply: oneshot::Sender<StoreResult<PageId>>,
    },
    UpsertPageBatch {
        pages: Vec<NewPage>,
        reply: oneshot::Sender<StoreResult<Vec<PageId>>>,
    },
    BeginSession {
        session: NewSession,
        reply: oneshot::Sender<StoreResult<()>>,
    },
    EndSession {
        session_id: SessionId,
        summary_page_id: Option<PageId>,
        reply: oneshot::Sender<StoreResult<()>>,
    },
    InsertObservation {
        obs: NewObservation,
        reply: oneshot::Sender<StoreResult<ObservationId>>,
    },
    InsertHandoff {
        handoff: NewHandoff,
        reply: oneshot::Sender<StoreResult<HandoffId>>,
    },
    AcceptHandoff {
        handoff_id: HandoffId,
        accepting_agent: AgentKind,
        accepting_session: Option<SessionId>,
        reply: oneshot::Sender<StoreResult<()>>,
    },
    /// Retro-fit sessions + observations to per-cwd projects and graveyard
    /// mash-up pages. Executed in one transaction for atomicity.
    Reorg {
        /// Each entry is `(session_id, new_project_id)`.
        plan: Vec<(SessionId, ProjectId)>,
        reply: oneshot::Sender<StoreResult<ReorgSummary>>,
    },
    BumpAccess {
        page_ids: Vec<PageId>,
        reply: oneshot::Sender<StoreResult<()>>,
    },
    SoftDeleteForDecay {
        page_ids: Vec<PageId>,
        reply: oneshot::Sender<StoreResult<usize>>,
    },
    HardDeleteDecayed {
        hard_delete_after_days: i64,
        reply: oneshot::Sender<StoreResult<usize>>,
    },
    StoreEmbedding {
        page_id: PageId,
        vector_bytes: Vec<u8>,
        provider: String,
        model: String,
        dim: u32,
        reply: oneshot::Sender<StoreResult<()>>,
    },
    Shutdown,
}

/// Cheap, cloneable handle that submits commands to the writer.
#[derive(Clone)]
pub struct WriterHandle {
    inner: Arc<WriterInner>,
}

struct WriterInner {
    tx: mpsc::Sender<WriteCmd>,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl WriterHandle {
    /// Take ownership of `conn` and spawn the writer thread.
    pub(crate) fn spawn(conn: Connection) -> Self {
        let (tx, rx) = mpsc::channel(1024);
        let handle = thread::Builder::new()
            .name("ai-memory-writer".into())
            .spawn(move || worker_loop(conn, rx))
            .expect("spawn writer thread");

        Self {
            inner: Arc::new(WriterInner {
                tx,
                join: Mutex::new(Some(handle)),
            }),
        }
    }

    /// Resolve a workspace by name, creating it atomically if missing.
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] if the actor has shut down, or
    /// propagates the SQL error from [`ops::get_or_create_workspace`].
    pub async fn get_or_create_workspace(
        &self,
        name: impl Into<String>,
    ) -> StoreResult<WorkspaceId> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::GetOrCreateWorkspace {
            name: name.into(),
            reply: tx,
        })
        .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Resolve a project by `(workspace_id, name)`, creating it atomically
    /// if missing.
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] if the actor has shut down, or
    /// propagates the SQL error from [`ops::get_or_create_project`].
    pub async fn get_or_create_project(
        &self,
        workspace_id: WorkspaceId,
        name: impl Into<String>,
        repo_path: Option<String>,
    ) -> StoreResult<ProjectId> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::GetOrCreateProject {
            workspace_id,
            name: name.into(),
            repo_path,
            reply: tx,
        })
        .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Begin a session (idempotent on the supplied id).
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] or propagates SQL errors.
    pub async fn begin_session(&self, session: NewSession) -> StoreResult<()> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::BeginSession { session, reply: tx })
            .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Mark a session ended, optionally linking its summary page.
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] or propagates SQL errors.
    pub async fn end_session(
        &self,
        session_id: SessionId,
        summary_page_id: Option<PageId>,
    ) -> StoreResult<()> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::EndSession {
            session_id,
            summary_page_id,
            reply: tx,
        })
        .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Append an observation row.
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] or propagates SQL errors.
    pub async fn insert_observation(&self, obs: NewObservation) -> StoreResult<ObservationId> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::InsertObservation { obs, reply: tx })
            .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Insert a new handoff in `open` state.
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] or propagates SQL errors.
    pub async fn insert_handoff(&self, handoff: NewHandoff) -> StoreResult<HandoffId> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::InsertHandoff { handoff, reply: tx })
            .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Mark a handoff accepted by the given agent / session.
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] or propagates SQL errors.
    pub async fn accept_handoff(
        &self,
        handoff_id: HandoffId,
        accepting_agent: AgentKind,
        accepting_session: Option<SessionId>,
    ) -> StoreResult<()> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::AcceptHandoff {
            handoff_id,
            accepting_agent,
            accepting_session,
            reply: tx,
        })
        .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Store (or replace) the embedding for one page (M9).
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] or propagates SQL errors.
    pub async fn store_embedding(
        &self,
        page_id: PageId,
        vector_bytes: Vec<u8>,
        provider: String,
        model: String,
        dim: u32,
    ) -> StoreResult<()> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::StoreEmbedding {
            page_id,
            vector_bytes,
            provider,
            model,
            dim,
            reply: tx,
        })
        .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Bump access counters for a set of pages (M8 reinforcement term).
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] or propagates SQL errors.
    pub async fn bump_access(&self, page_ids: Vec<PageId>) -> StoreResult<()> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::BumpAccess {
            page_ids,
            reply: tx,
        })
        .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Soft-delete pages identified by the M8 forget sweep.
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] or propagates SQL errors.
    pub async fn soft_delete_for_decay(&self, page_ids: Vec<PageId>) -> StoreResult<usize> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::SoftDeleteForDecay {
            page_ids,
            reply: tx,
        })
        .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Hard-delete pages soft-deleted by the sweep more than
    /// `hard_delete_after_days` ago.
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] or propagates SQL errors.
    pub async fn hard_delete_decayed(&self, hard_delete_after_days: i64) -> StoreResult<usize> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::HardDeleteDecayed {
            hard_delete_after_days,
            reply: tx,
        })
        .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Retro-fit sessions and their observations to per-cwd projects and
    /// graveyard any mash-up pages. The `plan` slice contains
    /// `(session_id, new_project_id)` pairs. Everything runs in one
    /// SQLite transaction — either fully committed or fully rolled back.
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] if the actor has shut down, or
    /// propagates the SQL error from the reorg transaction.
    pub async fn reorg_sessions(
        &self,
        plan: Vec<(SessionId, ProjectId)>,
    ) -> StoreResult<ReorgSummary> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::Reorg { plan, reply: tx }).await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Upsert a batch of pages atomically (one SQL transaction).
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] if the actor has shut down,
    /// or propagates the SQL error from [`ops::upsert_pages_batch`].
    pub async fn upsert_pages_batch(&self, pages: Vec<NewPage>) -> StoreResult<Vec<PageId>> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::UpsertPageBatch { pages, reply: tx })
            .await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    /// Upsert a page (creating it or superseding the existing latest
    /// version when the body has changed).
    ///
    /// # Errors
    /// Returns [`StoreError::WriterClosed`] if the actor has shut down, or
    /// propagates the SQL error from [`ops::upsert_page`].
    pub async fn upsert_page(&self, page: NewPage) -> StoreResult<PageId> {
        let (tx, rx) = oneshot::channel();
        self.send(WriteCmd::UpsertPage { page, reply: tx }).await?;
        rx.await.map_err(|_| StoreError::WriterClosed)?
    }

    async fn send(&self, cmd: WriteCmd) -> StoreResult<()> {
        self.inner
            .tx
            .send(cmd)
            .await
            .map_err(|_| StoreError::WriterClosed)
    }
}

impl Drop for WriterInner {
    fn drop(&mut self) {
        let _ = self.tx.try_send(WriteCmd::Shutdown);
        if let Some(handle) = self.join.lock().expect("writer join mutex poisoned").take() {
            let _ = handle.join();
        }
    }
}

/// Dispatch one operation's result back to its caller. Logs a warn
/// when the receiver was dropped (caller cancelled their `.await` or
/// hit a timeout) so the operator sees backpressure / cancellation
/// noise instead of silent loss. The result itself is consumed by
/// the failed `send` and discarded — the caller's await has already
/// returned a JoinError-shaped failure by this point.
fn send_or_warn<T>(reply: oneshot::Sender<T>, result: T, op: &'static str) {
    if reply.send(result).is_err() {
        tracing::warn!(
            op,
            "writer reply dropped — caller cancelled or oneshot receiver closed"
        );
    }
}

fn worker_loop(mut conn: Connection, mut rx: mpsc::Receiver<WriteCmd>) {
    while let Some(cmd) = rx.blocking_recv() {
        match cmd {
            WriteCmd::Shutdown => break,
            WriteCmd::GetOrCreateWorkspace { name, reply } => {
                let result = ops::get_or_create_workspace(&mut conn, &name);
                send_or_warn(reply, result, "get_or_create_workspace");
            }
            WriteCmd::GetOrCreateProject {
                workspace_id,
                name,
                repo_path,
                reply,
            } => {
                let result = ops::get_or_create_project(
                    &mut conn,
                    &workspace_id,
                    &name,
                    repo_path.as_deref(),
                );
                send_or_warn(reply, result, "get_or_create_project");
            }
            WriteCmd::UpsertPage { page, reply } => {
                let result = ops::upsert_page(&mut conn, &page);
                send_or_warn(reply, result, "upsert_page");
            }
            WriteCmd::UpsertPageBatch { pages, reply } => {
                let result = ops::upsert_pages_batch(&mut conn, &pages);
                send_or_warn(reply, result, "upsert_pages_batch");
            }
            WriteCmd::BeginSession { session, reply } => {
                let result = ops::begin_session(&mut conn, &session);
                send_or_warn(reply, result, "begin_session");
            }
            WriteCmd::EndSession {
                session_id,
                summary_page_id,
                reply,
            } => {
                let result = ops::end_session(&mut conn, &session_id, summary_page_id.as_ref());
                send_or_warn(reply, result, "end_session");
            }
            WriteCmd::InsertObservation { obs, reply } => {
                let result = ops::insert_observation(&mut conn, &obs);
                send_or_warn(reply, result, "insert_observation");
            }
            WriteCmd::InsertHandoff { handoff, reply } => {
                let result = ops::insert_handoff(&mut conn, &handoff);
                send_or_warn(reply, result, "insert_handoff");
            }
            WriteCmd::AcceptHandoff {
                handoff_id,
                accepting_agent,
                accepting_session,
                reply,
            } => {
                let result = ops::accept_handoff(
                    &mut conn,
                    &handoff_id,
                    accepting_agent,
                    accepting_session.as_ref(),
                );
                send_or_warn(reply, result, "accept_handoff");
            }
            WriteCmd::Reorg { plan, reply } => {
                let result = ops::reorg_sessions(&mut conn, &plan);
                send_or_warn(reply, result, "reorg_sessions");
            }
            WriteCmd::BumpAccess { page_ids, reply } => {
                let result = ops::bump_access_for_pages(&mut conn, &page_ids);
                send_or_warn(reply, result, "bump_access_for_pages");
            }
            WriteCmd::SoftDeleteForDecay { page_ids, reply } => {
                let result = ops::soft_delete_for_decay(&mut conn, &page_ids);
                send_or_warn(reply, result, "soft_delete_for_decay");
            }
            WriteCmd::HardDeleteDecayed {
                hard_delete_after_days,
                reply,
            } => {
                let result = ops::hard_delete_decayed_pages(&mut conn, hard_delete_after_days);
                send_or_warn(reply, result, "hard_delete_decayed_pages");
            }
            WriteCmd::StoreEmbedding {
                page_id,
                vector_bytes,
                provider,
                model,
                dim,
                reply,
            } => {
                let result = ops::store_embedding(
                    &mut conn,
                    &page_id,
                    &vector_bytes,
                    &provider,
                    &model,
                    dim,
                );
                send_or_warn(reply, result, "store_embedding");
            }
        }
    }
    tracing::debug!("writer thread exiting cleanly");
}
