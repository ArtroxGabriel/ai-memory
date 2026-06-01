//! Integration tests for `GET /admin/read-page`, focused on the DB-backed
//! fallback: when the on-disk markdown read fails (index ahead of disk), the
//! handler serves the store's faithful copy instead of 404ing
//! (gotchas/read-page-by-query-misses).

use ai_memory_core::{NewPage, PagePath, Tier};
use ai_memory_mcp::{AdminState, admin_router};
use ai_memory_store::{DecayParams, Store};
use ai_memory_wiki::{Wiki, WritePageRequest};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tempfile::TempDir;
use tower::ServiceExt;

async fn make_state(tmp: &TempDir) -> (AdminState, Store) {
    let store = Store::open(tmp.path()).unwrap();
    let wiki = Wiki::new(tmp.path(), store.writer.clone()).unwrap();
    let db_path = store.db_path().to_path_buf();
    let state = AdminState {
        writer: store.writer.clone(),
        reader: store.reader.clone(),
        wiki,
        llm: None,
        embedder: None,
        provider_health: ai_memory_llm::ProviderHealth::default(),
        decay_params: DecayParams::default(),
        data_dir: tmp.path().to_path_buf(),
        db_path,
        bind: "127.0.0.1:49374".to_string(),
        bootstrap_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        token_pepper: None,
    };
    (state, store)
}

async fn body_json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
}

async fn get(state: AdminState, uri: &str) -> axum::response::Response {
    let router = admin_router(state);
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    router.oneshot(req).await.unwrap()
}

/// A page present in the store but NOT on disk (the index is ahead of the
/// filesystem) is still served — from the DB copy — rather than 404ing.
#[tokio::test]
async fn read_page_falls_back_to_db_when_file_missing() {
    let tmp = TempDir::new().unwrap();
    let (state, store) = make_state(&tmp).await;

    // Seed straight through the store writer: this writes the DB row + FTS
    // index but NEVER touches disk, so the on-disk read will fail.
    let ws = store
        .writer
        .get_or_create_workspace("default".to_string())
        .await
        .unwrap();
    let proj = store
        .writer
        .get_or_create_project(ws, "scratch".to_string(), None)
        .await
        .unwrap();
    store
        .writer
        .upsert_page(NewPage {
            workspace_id: ws,
            project_id: proj,
            path: PagePath::new("notes/db-only.md").unwrap(),
            title: "DB Only".to_string(),
            body: "this body lives only in the database".to_string(),
            tier: Tier::Semantic,
            frontmatter_json: serde_json::json!({"title": "DB Only"}),
            pinned: false,
            links: Vec::new(),
            author_id: None,
        })
        .await
        .unwrap();

    let resp = get(
        state,
        "/admin/read-page?workspace=default&project=scratch&path=notes%2Fdb-only.md",
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK, "must serve from DB fallback");
    let body = body_json(resp).await;
    assert_eq!(
        body["body"].as_str().unwrap_or(""),
        "this body lives only in the database",
        "{body}"
    );
    assert_eq!(body["title"], "DB Only", "{body}");
}

/// Sanity: a page written through the wiki (disk + index) is served from disk
/// and reads back its body. Guards the happy path the fallback wraps.
#[tokio::test]
async fn read_page_serves_on_disk_page() {
    let tmp = TempDir::new().unwrap();
    let (state, store) = make_state(&tmp).await;
    let ws = store
        .writer
        .get_or_create_workspace("default".to_string())
        .await
        .unwrap();
    let proj = store
        .writer
        .get_or_create_project(ws, "scratch".to_string(), None)
        .await
        .unwrap();
    state
        .wiki
        .write_page(WritePageRequest {
            workspace_id: ws,
            project_id: proj,
            path: PagePath::new("notes/on-disk.md").unwrap(),
            frontmatter: serde_json::json!({"title": "On Disk"}),
            body: "written to disk via the wiki".to_string(),
            tier: Tier::Semantic,
            pinned: false,
            title: Some("On Disk".into()),
            author_id: None,
            actor: ai_memory_core::ActorContext::anonymous(),
        })
        .await
        .unwrap();

    let resp = get(
        state,
        "/admin/read-page?workspace=default&project=scratch&path=notes%2Fon-disk.md",
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp).await;
    assert_eq!(
        body["body"].as_str().unwrap_or(""),
        "written to disk via the wiki",
        "{body}"
    );
}

/// A genuinely-absent page (no DB row, no file) still 404s — the fallback
/// must not mask a real miss.
#[tokio::test]
async fn read_page_404_when_truly_absent() {
    let tmp = TempDir::new().unwrap();
    let (state, store) = make_state(&tmp).await;
    // Create the workspace/project so resolution succeeds but the page does not.
    let ws = store
        .writer
        .get_or_create_workspace("default".to_string())
        .await
        .unwrap();
    store
        .writer
        .get_or_create_project(ws, "scratch".to_string(), None)
        .await
        .unwrap();

    let resp = get(
        state,
        "/admin/read-page?workspace=default&project=scratch&path=notes%2Fnope.md",
    )
    .await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
