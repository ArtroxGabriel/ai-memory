//! SQLite storage layer for ai-memory.
//!
//! The crate owns a single SQLite file under `<data_dir>/db/memory.sqlite`,
//! opens it in WAL mode with foreign keys on, runs all pending migrations
//! at startup, and exposes a [`WriterHandle`] that serialises every mutation
//! through a dedicated OS thread.
//!
//! Reader-side APIs land in milestone M1-B; the writer + migrations are
//! sufficient for M1-A's "drop a page in, see it persisted" demo.

use std::path::{Path, PathBuf};

use rusqlite::Connection;

mod error;
mod migrations;
mod ops;
mod reader;
mod writer;

pub use error::{StoreError, StoreResult};
pub use reader::{PageHit, ReaderPool, StatusCounts};
pub use writer::WriterHandle;

/// Filename used inside the data dir's `db/` subdirectory.
pub const DB_FILENAME: &str = "memory.sqlite";

/// Default soft cap for the read-only connection pool.
const READER_POOL_SOFT_CAP: usize = 4;

/// Open (and migrate) a [`Store`] rooted at the given data directory.
pub struct Store {
    /// Cloneable handle to submit mutations.
    pub writer: WriterHandle,
    /// Cloneable handle for read-only queries.
    pub reader: ReaderPool,
    db_path: PathBuf,
}

impl Store {
    /// Open the SQLite file at `<data_dir>/db/memory.sqlite`, applying any
    /// outstanding migrations, then spawn the writer thread and prepare
    /// the read-only connection pool.
    ///
    /// # Errors
    /// Returns [`StoreError`] if the file cannot be opened, migrations
    /// cannot be applied, or the writer thread fails to start.
    pub fn open(data_dir: &Path) -> StoreResult<Self> {
        let db_dir = data_dir.join("db");
        std::fs::create_dir_all(&db_dir)?;
        let db_path = db_dir.join(DB_FILENAME);

        let mut conn = Connection::open(&db_path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.pragma_update(None, "busy_timeout", 5_000)?; // ms

        migrations::run(&mut conn)?;

        let writer = WriterHandle::spawn(conn);
        let reader = ReaderPool::new(&db_path, READER_POOL_SOFT_CAP)?;
        Ok(Self {
            writer,
            reader,
            db_path,
        })
    }

    /// Path of the SQLite file on disk.
    #[must_use]
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory_core::{NewPage, PagePath, ProjectId, Tier, WorkspaceId};
    use tempfile::TempDir;

    fn sample_page(ws: WorkspaceId, proj: ProjectId, path: &str, body: &str) -> NewPage {
        NewPage {
            workspace_id: ws,
            project_id: proj,
            path: PagePath::new(path).unwrap(),
            title: "test".into(),
            body: body.into(),
            tier: Tier::Semantic,
            frontmatter_json: serde_json::json!({}),
            pinned: false,
        }
    }

    #[tokio::test]
    async fn open_and_upsert_page() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "ai-memory", None)
            .await
            .unwrap();
        let id_a = store
            .writer
            .upsert_page(sample_page(ws, proj, "foo.md", "hello"))
            .await
            .unwrap();
        // Same body again: returns the same id, no supersession.
        let id_b = store
            .writer
            .upsert_page(sample_page(ws, proj, "foo.md", "hello"))
            .await
            .unwrap();
        assert_eq!(id_a, id_b);
        // Different body: supersession produces a new id.
        let id_c = store
            .writer
            .upsert_page(sample_page(ws, proj, "foo.md", "hello world"))
            .await
            .unwrap();
        assert_ne!(id_b, id_c);
    }

    #[tokio::test]
    async fn get_or_create_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let a = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let b = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        assert_eq!(a, b);
        let pa = store
            .writer
            .get_or_create_project(a, "scratch", None)
            .await
            .unwrap();
        let pb = store
            .writer
            .get_or_create_project(a, "scratch", None)
            .await
            .unwrap();
        assert_eq!(pa, pb);
    }

    #[tokio::test]
    async fn serialises_parallel_writes() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "ai-memory", None)
            .await
            .unwrap();
        // Spawn 16 concurrent writes; the writer must serialise them.
        let mut handles = Vec::new();
        for i in 0..16 {
            let w = store.writer.clone();
            handles.push(tokio::spawn(async move {
                w.upsert_page(sample_page(
                    ws,
                    proj,
                    &format!("p{i}.md"),
                    &format!("body-{i}"),
                ))
                .await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
    }

    #[tokio::test]
    async fn recent_pages_returns_latest_only_in_order() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "scratch", None)
            .await
            .unwrap();
        for i in 0..3u8 {
            store
                .writer
                .upsert_page(sample_page(
                    ws,
                    proj,
                    &format!("p{i}.md"),
                    &format!("body-{i}"),
                ))
                .await
                .unwrap();
        }
        // Bump the second page to force a later updated_at.
        store
            .writer
            .upsert_page(sample_page(ws, proj, "p1.md", "body-1-rev"))
            .await
            .unwrap();

        let hits = store.reader.recent_pages(10).await.unwrap();
        assert_eq!(hits.len(), 3, "is_latest filter should give us 3 pages");
        assert_eq!(
            hits[0].path.as_str(),
            "p1.md",
            "most-recently-updated first"
        );
    }

    #[tokio::test]
    async fn status_counts_zero_on_fresh_db() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let counts = store.reader.status_counts().await.unwrap();
        assert_eq!(counts.pages_latest, 0);
        assert_eq!(counts.pages_all, 0);
        assert_eq!(counts.sessions, 0);
        assert_eq!(counts.observations, 0);
    }

    #[tokio::test]
    async fn search_finds_inserted_page_and_counts_reflect_supersession() {
        let tmp = TempDir::new().unwrap();
        let store = Store::open(tmp.path()).unwrap();
        let ws = store
            .writer
            .get_or_create_workspace("default")
            .await
            .unwrap();
        let proj = store
            .writer
            .get_or_create_project(ws, "ai-memory", None)
            .await
            .unwrap();

        store
            .writer
            .upsert_page(sample_page(
                ws,
                proj,
                "alpha.md",
                "the quick brown fox jumps over the lazy dog",
            ))
            .await
            .unwrap();

        let hits = store.reader.search_pages("quick".into(), 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path.as_str(), "alpha.md");
        assert!(hits[0].snippet.contains("<mark>quick</mark>"));

        // Supersede: only the latest version should appear in counts'
        // pages_latest, and search should still return exactly one hit.
        store
            .writer
            .upsert_page(sample_page(
                ws,
                proj,
                "alpha.md",
                "a different sentence with quick still inside",
            ))
            .await
            .unwrap();

        let counts = store.reader.status_counts().await.unwrap();
        assert_eq!(counts.pages_latest, 1);
        assert_eq!(counts.pages_all, 2);

        let hits = store.reader.search_pages("quick".into(), 10).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0].snippet.contains("different"),
            "snippet should come from the latest version, got: {}",
            hits[0].snippet
        );
    }
}
