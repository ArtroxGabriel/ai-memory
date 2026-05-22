//! [`Wiki`] — the only correct write path for the markdown source-of-truth.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ai_memory_core::{NewPage, PageId, PagePath, ProjectId, Tier, WorkspaceId};
use ai_memory_llm::Embedder;
use ai_memory_store::{WriterHandle, f32_vec_to_bytes};

use crate::atomic;
use crate::error::WikiResult;
use crate::git::GitAdapter;
use crate::markdown::{Markdown, derive_title, emit, parse};

/// Wiki filesystem handle.
///
/// Owns the path of the wiki root (`<data_dir>/wiki/`) and a cloneable
/// [`WriterHandle`] so that every public mutation writes the markdown
/// file *and* sends a `WriteCmd::UpsertPage` to the store in a single
/// call — no background-task indexing-after-return (basic-memory #763
/// lesson).
#[derive(Clone)]
pub struct Wiki {
    root: PathBuf,
    writer: WriterHandle,
    git: GitAdapter,
    embedder: Option<Arc<dyn Embedder>>,
}

impl Wiki {
    /// Construct a wiki handle rooted at `<data_dir>/wiki/`. Creates the
    /// directory if absent and initialises a git repo inside it.
    ///
    /// # Errors
    /// Returns [`WikiError::Io`] if the wiki root or git repo cannot be
    /// created.
    pub fn new(data_dir: &Path, writer: WriterHandle) -> WikiResult<Self> {
        let root = data_dir.join("wiki");
        std::fs::create_dir_all(&root)?;
        let git = GitAdapter::open_or_init(&root)?;
        Ok(Self {
            root,
            writer,
            git,
            embedder: None,
        })
    }

    /// Attach an embedder. When set, every successful `write_page` /
    /// `apply_batch` also computes + stores an embedding for the new
    /// version. Without this, vector search is unavailable and
    /// `ReaderPool::hybrid_search` falls back to pure FTS5.
    #[must_use]
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Borrow the optional embedder (used by the `ai-memory embed`
    /// backfill command).
    #[must_use]
    pub fn embedder(&self) -> Option<&Arc<dyn Embedder>> {
        self.embedder.as_ref()
    }

    /// Borrow the git adapter (for callers wiring auto-commit).
    #[must_use]
    pub fn git(&self) -> &GitAdapter {
        &self.git
    }

    /// Stage + commit the entire wiki tree. Returns `Ok(None)` if there
    /// was nothing to commit.
    ///
    /// # Errors
    /// Propagates [`WikiError`] from the git adapter.
    pub fn commit_all(&self, message: &str) -> WikiResult<Option<git2::Oid>> {
        self.git.commit_all(message)
    }

    /// Path of the wiki root on disk.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Absolute on-disk path for a given [`PagePath`].
    #[must_use]
    pub fn abs_path(&self, path: &PagePath) -> PathBuf {
        self.root.join(path.as_str())
    }

    /// Read the page at `path` from disk.
    ///
    /// # Errors
    /// Returns [`WikiError::Io`] if the file is missing or unreadable, or
    /// [`WikiError::Yaml`] if the frontmatter block is malformed.
    pub fn read_page(&self, path: &PagePath) -> WikiResult<Markdown> {
        let abs = self.abs_path(path);
        let raw = std::fs::read_to_string(&abs)?;
        parse(&raw)
    }

    /// Cloneable handle to the underlying store writer.
    #[must_use]
    pub fn writer(&self) -> &WriterHandle {
        &self.writer
    }

    /// Re-index the page on disk at `path` into the store *without*
    /// rewriting the file.
    ///
    /// Called by the watcher when an external editor (Obsidian, vim) has
    /// changed a file we did not write. The store-side sha256 short-circuit
    /// makes this idempotent: if the on-disk content already matches the
    /// latest version, no supersession happens.
    ///
    /// # Errors
    /// Returns [`WikiError`] for any filesystem, parsing, or store error.
    pub async fn reindex_page(
        &self,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
        path: PagePath,
    ) -> WikiResult<PageId> {
        let md = self.read_page(&path)?;
        let title = derive_title(&md.frontmatter, &md.body, &path);
        let id = self
            .writer
            .upsert_page(NewPage {
                workspace_id,
                project_id,
                path,
                title,
                body: md.body,
                tier: Tier::Semantic,
                frontmatter_json: md.frontmatter,
                pinned: false,
            })
            .await?;
        Ok(id)
    }

    /// Atomically apply a batch of page writes. Either all pages land
    /// (one SQL transaction) and their files are renamed into place,
    /// or no DB row changes and tempfiles are dropped.
    ///
    /// # Errors
    /// Returns [`WikiError`] for any filesystem, parsing, or store
    /// error.
    pub async fn apply_batch(&self, requests: Vec<WritePageRequest>) -> WikiResult<Vec<PageId>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        // Pre-compute markdown + tempfile for each request.
        let mut staged: Vec<(
            WritePageRequest,
            tempfile::NamedTempFile,
            std::path::PathBuf,
        )> = Vec::with_capacity(requests.len());
        for req in requests {
            let title = derive_title(&req.frontmatter, &req.body, &req.path);
            let markdown = Markdown {
                frontmatter: req.frontmatter.clone(),
                body: req.body.clone(),
            };
            let emitted = emit(&markdown)?;
            let abs = self.abs_path(&req.path);
            let parent = abs.parent().ok_or_else(|| {
                ai_memory_wiki_error("page path has no parent (cannot stage tempfile)")
            })?;
            std::fs::create_dir_all(parent)?;
            let mut tmp = tempfile::Builder::new()
                .prefix(".ai-memory-tmp.")
                .tempfile_in(parent)?;
            use std::io::Write as _;
            tmp.write_all(emitted.as_bytes())?;
            tmp.as_file().sync_data()?;
            let req_with_title = WritePageRequest {
                title: Some(title),
                ..req
            };
            staged.push((req_with_title, tmp, abs));
        }

        // Build NewPage batch with the precomputed titles.
        let pages: Vec<ai_memory_core::NewPage> = staged
            .iter()
            .map(|(req, _, _)| ai_memory_core::NewPage {
                workspace_id: req.workspace_id,
                project_id: req.project_id,
                path: req.path.clone(),
                title: req.title.clone().unwrap_or_default(),
                body: req.body.clone(),
                tier: req.tier,
                frontmatter_json: req.frontmatter.clone(),
                pinned: req.pinned,
            })
            .collect();

        let ids = self.writer.upsert_pages_batch(pages).await?;

        // SQL succeeded; rename tempfiles into place.
        for (_, tmp, abs) in staged {
            let persisted = tmp.persist(&abs)?;
            persisted.sync_data()?;
        }

        Ok(ids)
    }

    /// Write `body` (with optional `frontmatter`) atomically to
    /// `<wiki_root>/<path>` and upsert the matching page row in the store.
    ///
    /// The store side does the sha256 short-circuit + supersession dance.
    /// Returns the id of the page version that is now `is_latest = 1`.
    ///
    /// # Errors
    /// Returns [`WikiError`] for any filesystem, parsing, or store error.
    pub async fn write_page(&self, req: WritePageRequest) -> WikiResult<PageId> {
        let WritePageRequest {
            workspace_id,
            project_id,
            path,
            frontmatter,
            body,
            tier,
            pinned,
            title: explicit_title,
        } = req;

        let title = explicit_title
            .clone()
            .unwrap_or_else(|| derive_title(&frontmatter, &body, &path));
        let markdown = Markdown {
            frontmatter: frontmatter.clone(),
            body: body.clone(),
        };
        let emitted = emit(&markdown)?;
        let abs = self.abs_path(&path);
        atomic::write_atomic(&abs, emitted.as_bytes())?;

        let page_id = self
            .writer
            .upsert_page(NewPage {
                workspace_id,
                project_id,
                path,
                title,
                body: body.clone(),
                tier,
                frontmatter_json: frontmatter,
                pinned,
            })
            .await?;
        // Embed if configured. We do this on the caller's task so the
        // tool reply still happens "indexes commit in the same
        // transaction" (basic-memory #763 lesson): no fire-and-forget
        // background embedding.
        if let Some(embedder) = &self.embedder {
            match embedder.embed(&body).await {
                Ok(vec) => {
                    let bytes = f32_vec_to_bytes(&vec);
                    self.writer
                        .store_embedding(
                            page_id,
                            bytes,
                            embedder.provider().to_string(),
                            embedder.model().to_string(),
                            embedder.dim(),
                        )
                        .await?;
                }
                Err(e) => {
                    tracing::warn!(error = %e, path = %page_id, "embedding failed; page indexed without it");
                }
            }
        }
        Ok(page_id)
    }
}

/// Input bundle for [`Wiki::write_page`]. Carries the full 3-tuple
/// identity (`workspace_id`, `project_id`, `path`) plus body & metadata.
#[derive(Debug, Clone)]
pub struct WritePageRequest {
    /// Owning workspace.
    pub workspace_id: WorkspaceId,
    /// Owning project.
    pub project_id: ProjectId,
    /// Relative wiki path.
    pub path: PagePath,
    /// Optional frontmatter (JSON object). May be `Null` for no frontmatter.
    pub frontmatter: serde_json::Value,
    /// Markdown body (excluding any frontmatter block).
    pub body: String,
    /// Tier classification.
    pub tier: Tier,
    /// `true` if the user has pinned this page.
    pub pinned: bool,
    /// Optional pre-derived title (used by `apply_batch` to share the
    /// title between the staged markdown file + the store row).
    #[doc(hidden)]
    pub title: Option<String>,
}

fn ai_memory_wiki_error(msg: &str) -> crate::WikiError {
    crate::WikiError::Io(std::io::Error::other(msg.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory_store::Store;
    use tempfile::TempDir;

    fn req(
        ws: WorkspaceId,
        proj: ProjectId,
        path: &str,
        body: &str,
        fm: serde_json::Value,
    ) -> WritePageRequest {
        WritePageRequest {
            workspace_id: ws,
            project_id: proj,
            path: PagePath::new(path).unwrap(),
            frontmatter: fm,
            body: body.into(),
            tier: Tier::Semantic,
            pinned: false,
            title: None,
        }
    }

    #[tokio::test]
    async fn write_page_writes_file_and_indexes_in_store() {
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
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();

        let id = wiki
            .write_page(req(
                ws,
                proj,
                "notes/karpathy.md",
                "Karpathy says: compile, do not retrieve.\n",
                serde_json::json!({ "title": "Karpathy LLM Wiki" }),
            ))
            .await
            .unwrap();
        let _ = id; // any non-zero PageId is sufficient

        // File is on disk with the frontmatter back.
        let on_disk = std::fs::read_to_string(tmp.path().join("wiki/notes/karpathy.md")).unwrap();
        assert!(on_disk.starts_with("---\n"));
        assert!(on_disk.contains("title: Karpathy LLM Wiki"));
        assert!(on_disk.contains("Karpathy says"));

        // FTS5 finds it via the store reader.
        let hits = store
            .reader
            .search_pages("karpathy".into(), 5)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Karpathy LLM Wiki");
        assert!(hits[0].snippet.contains("compile"));
    }

    #[tokio::test]
    async fn apply_batch_persists_all_pages_in_one_transaction() {
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
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
        let batch: Vec<_> = (0..5)
            .map(|i| WritePageRequest {
                workspace_id: ws,
                project_id: proj,
                path: PagePath::new(format!("batch/{i}.md")).unwrap(),
                frontmatter: serde_json::json!({"title": format!("Page {i}")}),
                body: format!("batch page {i} body line"),
                tier: Tier::Semantic,
                pinned: false,
                title: None,
            })
            .collect();
        let ids = wiki.apply_batch(batch).await.unwrap();
        assert_eq!(ids.len(), 5);
        for i in 0..5 {
            let path = tmp.path().join(format!("wiki/batch/{i}.md"));
            assert!(path.is_file(), "missing file {i}");
            let body = std::fs::read_to_string(&path).unwrap();
            assert!(body.contains(&format!("Page {i}")));
        }
        let counts = store.reader.status_counts().await.unwrap();
        assert_eq!(counts.pages_latest, 5);
        let hits = store.reader.search_pages("batch".into(), 10).await.unwrap();
        assert_eq!(hits.len(), 5);
    }

    #[tokio::test]
    async fn rewriting_same_body_is_idempotent() {
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
        let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();

        let r = |body: &str| req(ws, proj, "a.md", body, serde_json::json!({ "title": "A" }));

        let a = wiki.write_page(r("body one")).await.unwrap();
        let b = wiki.write_page(r("body one")).await.unwrap();
        assert_eq!(a, b);
        let c = wiki.write_page(r("body two")).await.unwrap();
        assert_ne!(b, c);
    }
}
