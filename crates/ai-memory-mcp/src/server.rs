//! [`AiMemoryServer`] — the MCP server skeleton + tool router.

use ai_memory_store::ReaderPool;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, Content, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo,
};
use rmcp::{ErrorData as McpError, ServerHandler, schemars, tool, tool_handler, tool_router};
use serde::{Deserialize, Serialize};

/// Instructions surfaced to clients via `ServerInfo`. Short and
/// agent-readable — Claude Code / Codex will see this in their session
/// preamble.
pub const MEMORY_INSTRUCTIONS: &str = "Long-term memory for coding agents. Use \
memory_query for free-text search, memory_recent to peek at recently-changed \
pages, and memory_status for counts. All tools are read-only; writes happen \
automatically via hooks and the watcher.";

/// MCP server backed by the ai-memory store.
#[derive(Clone)]
pub struct AiMemoryServer {
    reader: ReaderPool,
    default_limit: usize,
    // Read by the `#[tool_handler]` macro expansion; rustc's dead-code
    // analysis can't see that, so the lint must be allowed explicitly.
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct QueryArgs {
    /// FTS5 query expression (e.g. `"karpathy wiki"` or `quick OR slow`).
    #[serde(alias = "q", alias = "search")]
    query: String,
    /// Maximum number of hits to return (default 10, max 100).
    #[serde(default, alias = "n", alias = "top_k")]
    limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct RecentArgs {
    /// Maximum number of recent pages to return (default 10, max 100).
    #[serde(default, alias = "n")]
    limit: Option<usize>,
}

#[derive(Debug, Serialize)]
struct QueryResponse<T: Serialize> {
    hits: Vec<T>,
}

#[derive(Debug, Serialize)]
struct StatusResponse {
    counts: ai_memory_store::StatusCounts,
}

#[tool_router]
impl AiMemoryServer {
    /// Construct a server backed by the given reader pool.
    #[must_use]
    pub fn new(reader: ReaderPool) -> Self {
        Self {
            reader,
            default_limit: 10,
            tool_router: Self::tool_router(),
        }
    }

    /// Full-text search the wiki via FTS5. Returns up to `limit` hits with
    /// HTML-marked snippets and a rank score.
    #[tool(description = "Full-text search the long-term memory wiki via FTS5. \
        Returns up to `limit` matching pages with HTML-marked snippets and a \
        rank score (lower rank = better match). Only the latest version of \
        each page is searched.")]
    async fn memory_query(
        &self,
        Parameters(args): Parameters<QueryArgs>,
    ) -> Result<CallToolResult, McpError> {
        let limit = args.limit.unwrap_or(self.default_limit).clamp(1, 100);
        let hits = self
            .reader
            .search_pages(args.query, limit)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let response = QueryResponse { hits };
        ok_json(&response)
    }

    /// Return the N most-recently-updated pages.
    #[tool(description = "Return the N most-recently-updated wiki pages \
        (descending by updated_at). Useful for resuming a session: \
        the agent can read the last few pages to see what was worked on.")]
    async fn memory_recent(
        &self,
        Parameters(args): Parameters<RecentArgs>,
    ) -> Result<CallToolResult, McpError> {
        let limit = args.limit.unwrap_or(self.default_limit).clamp(1, 100);
        let hits = self
            .reader
            .recent_pages(limit)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let response = QueryResponse { hits };
        ok_json(&response)
    }

    /// Report aggregate counts (pages, sessions, observations).
    #[tool(description = "Report aggregate memory counts and runtime status \
        (pages latest, pages all versions, sessions, observations). \
        Use this at session start to see how much context the agent has \
        accumulated for this workspace.")]
    async fn memory_status(&self) -> Result<CallToolResult, McpError> {
        let counts = self
            .reader
            .status_counts()
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let response = StatusResponse { counts };
        ok_json(&response)
    }
}

#[tool_handler]
impl ServerHandler for AiMemoryServer {
    fn get_info(&self) -> ServerInfo {
        // `Implementation::from_build_env()` reads CARGO_PKG_NAME/VERSION
        // from *rmcp's* compilation unit, not ours. Patch the fields
        // post-construction so the wire protocol surfaces "ai-memory".
        let mut implementation = Implementation::from_build_env();
        implementation.name = "ai-memory".into();
        implementation.version = env!("CARGO_PKG_VERSION").into();
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(implementation)
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions(MEMORY_INSTRUCTIONS.to_string())
    }
}

fn ok_json<T: Serialize>(value: &T) -> Result<CallToolResult, McpError> {
    let s = serde_json::to_string_pretty(value)
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
    Ok(CallToolResult::success(vec![Content::text(s)]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_memory_core::{NewPage, PagePath, Tier};
    use ai_memory_store::Store;
    use tempfile::TempDir;

    async fn setup_server() -> (TempDir, Store, AiMemoryServer) {
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
        store
            .writer
            .upsert_page(NewPage {
                workspace_id: ws,
                project_id: proj,
                path: PagePath::new("foo.md").unwrap(),
                title: "Foo".into(),
                body: "Karpathy says compile, not retrieve.".into(),
                tier: Tier::Semantic,
                frontmatter_json: serde_json::json!({}),
                pinned: false,
            })
            .await
            .unwrap();

        let server = AiMemoryServer::new(store.reader.clone());
        (tmp, store, server)
    }

    #[tokio::test]
    async fn server_constructs_with_tool_router() {
        let (_tmp, _store, _server) = setup_server().await;
    }

    #[tokio::test]
    async fn memory_query_returns_hits_via_tool_method() {
        let (_tmp, _store, server) = setup_server().await;
        let result = server
            .memory_query(Parameters(QueryArgs {
                query: "karpathy".into(),
                limit: Some(5),
            }))
            .await
            .unwrap();
        let text = match result.content.first().and_then(|c| c.as_text()) {
            Some(t) => t.text.clone(),
            None => panic!("expected text content"),
        };
        assert!(text.contains("foo.md"), "expected hit; got {text}");
    }

    #[tokio::test]
    async fn memory_status_returns_counts() {
        let (_tmp, _store, server) = setup_server().await;
        let result = server.memory_status().await.unwrap();
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.clone())
            .unwrap();
        assert!(text.contains("\"pages_latest\": 1"));
    }

    #[tokio::test]
    async fn memory_recent_returns_one_hit() {
        let (_tmp, _store, server) = setup_server().await;
        let result = server
            .memory_recent(Parameters(RecentArgs { limit: Some(5) }))
            .await
            .unwrap();
        let text = result
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.clone())
            .unwrap();
        assert!(text.contains("foo.md"), "expected hit; got {text}");
    }
}
