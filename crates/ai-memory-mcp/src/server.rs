//! [`AiMemoryServer`] — the MCP server skeleton + tool router.

use std::str::FromStr;
use std::sync::Arc;

use ai_memory_consolidate::{Consolidator, run_lint, run_sweep};
use ai_memory_core::{AgentKind, NewHandoff, PageId, ProjectId, SessionId, WorkspaceId};
use ai_memory_llm::{Embedder, LlmProvider};
use ai_memory_store::{DecayParams, ReaderPool, WriterHandle};
use ai_memory_wiki::Wiki;
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
    writer: WriterHandle,
    workspace_id: WorkspaceId,
    project_id: ProjectId,
    default_limit: usize,
    /// Optional LLM consolidator. When `None`, `memory_consolidate`
    /// returns a "not configured" error.
    consolidator: Option<Arc<Consolidator>>,
    /// Optional LLM provider for the lint contradiction pass. When
    /// `None`, lint runs only the rule-based checks.
    llm: Option<Arc<dyn LlmProvider>>,
    /// Wiki handle (needed by the sweep / lint tools to read pages +
    /// write the lint report). `None` when the server was built
    /// without one — older `new()` callers stay safe.
    wiki: Option<Wiki>,
    /// M8 retention parameters. Defaults if not overridden by the
    /// caller (typically from the user's config.toml `[decay]` block).
    decay_params: DecayParams,
    /// M9 embedder for hybrid query. When `None`, `memory_query`
    /// falls back to pure FTS5.
    embedder: Option<Arc<dyn Embedder>>,
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

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct SweepArgs {
    /// If true, preview only. Default false.
    #[serde(default)]
    dry_run: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct LintArgs {
    /// If true, don't write wiki/_lint/<date>.md. Default false.
    #[serde(default)]
    dry_run: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct ConsolidateArgs {
    /// UUID of the session to consolidate.
    session_id: String,
    /// If true, preview without writing. Default false.
    #[serde(default)]
    dry_run: Option<bool>,
    /// If true, M7b multi-page atomic fan-out. Default false (single page).
    #[serde(default)]
    multi_page: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct HandoffBeginArgs {
    /// Short prose summary of where the session left off.
    summary: String,
    /// Questions the next agent should resolve.
    #[serde(default)]
    open_questions: Vec<String>,
    /// Suggested next steps.
    #[serde(default)]
    next_steps: Vec<String>,
    /// Files touched during the session.
    #[serde(default)]
    files_touched: Vec<String>,
    /// Working directory at the time of handoff. Used to match the
    /// next agent's `memory_handoff_accept` call.
    #[serde(default)]
    cwd: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
struct HandoffAcceptArgs {
    /// Restrict the search to handoffs created for a specific cwd.
    #[serde(default)]
    cwd: Option<String>,
}

#[tool_router]
impl AiMemoryServer {
    /// Construct a server backed by the given reader/writer + 3-tuple
    /// identity coordinates.
    #[must_use]
    pub fn new(
        reader: ReaderPool,
        writer: WriterHandle,
        workspace_id: WorkspaceId,
        project_id: ProjectId,
    ) -> Self {
        Self {
            reader,
            writer,
            workspace_id,
            project_id,
            default_limit: 10,
            consolidator: None,
            llm: None,
            wiki: None,
            decay_params: DecayParams::default(),
            embedder: None,
            tool_router: Self::tool_router(),
        }
    }

    /// Attach an embedder for hybrid (FTS5 + vector RRF) query. Without
    /// this, `memory_query` runs pure FTS5.
    #[must_use]
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Override the retention-sweep parameters (typically populated
    /// from the user's config.toml `[decay]` table).
    #[must_use]
    pub fn with_decay_params(mut self, params: DecayParams) -> Self {
        self.decay_params = params;
        self
    }

    /// Attach the wiki handle. Without this, `memory_forget_sweep`
    /// and `memory_lint` cannot write their report pages.
    #[must_use]
    pub fn with_wiki(mut self, wiki: Wiki) -> Self {
        self.wiki = Some(wiki);
        self
    }

    /// Attach an LLM-backed consolidator. Without this, the
    /// `memory_consolidate` tool errors with "not configured". Also
    /// stores the LLM provider so `memory_lint` can run its
    /// contradiction pass.
    #[must_use]
    pub fn with_consolidator(mut self, wiki: Wiki, llm: Arc<dyn LlmProvider>) -> Self {
        let consolidator = Consolidator::new(
            self.reader.clone(),
            self.writer.clone(),
            wiki.clone(),
            llm.clone(),
            self.workspace_id,
            self.project_id,
        );
        self.consolidator = Some(Arc::new(consolidator));
        self.llm = Some(llm);
        self.wiki = Some(wiki);
        self
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
        let hits = if let Some(embedder) = &self.embedder {
            // Hybrid path: embed the query, RRF-fuse with FTS5.
            let qv = embedder
                .embed(&args.query)
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;
            self.reader
                .hybrid_search(
                    self.workspace_id,
                    self.project_id,
                    args.query,
                    Some(qv),
                    embedder.provider().to_string(),
                    embedder.model().to_string(),
                    embedder.dim(),
                    limit,
                )
                .await
        } else {
            self.reader.search_pages(args.query, limit).await
        };
        let hits = hits.map_err(|e| McpError::internal_error(e.to_string(), None))?;
        self.spawn_access_bump(hits.iter().map(|h| h.id).collect());
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
        self.spawn_access_bump(hits.iter().map(|h| h.id).collect());
        let response = QueryResponse { hits };
        ok_json(&response)
    }

    /// Run the M8 forget sweep over episodic pages.
    #[tool(description = "Run the retention sweep: walk is_latest=1 \
        episodic pages, score them with the agentmemory-style retention \
        formula (salience * exp(-lambda * age) + sigma * log(1 + accesses) \
        * exp(-mu * days_since_access)), and soft-delete those below the \
        cold threshold. Semantic / procedural / pinned pages are exempt. \
        Pass dry_run=true to preview.")]
    async fn memory_forget_sweep(
        &self,
        Parameters(args): Parameters<SweepArgs>,
    ) -> Result<CallToolResult, McpError> {
        let report = run_sweep(
            &self.reader,
            &self.writer,
            self.workspace_id,
            self.project_id,
            &self.decay_params,
            args.dry_run.unwrap_or(false),
        )
        .await
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        ok_json(&report)
    }

    /// Run the M8 lint pass: rule-based + optional LLM contradiction.
    #[tool(description = "Audit the wiki for stale episodic pages, \
        duplicate titles, broken cross-references, and (if an LLM \
        provider is configured) contradictions across semantic pages. \
        Findings land in wiki/_lint/<date>.md unless dry_run=true.")]
    async fn memory_lint(
        &self,
        Parameters(args): Parameters<LintArgs>,
    ) -> Result<CallToolResult, McpError> {
        let Some(wiki) = self.wiki.as_ref() else {
            return Err(McpError::internal_error(
                "memory_lint requires the server to be built with a wiki handle",
                None,
            ));
        };
        let report = run_lint(
            &self.reader,
            wiki,
            self.llm.as_ref(),
            self.workspace_id,
            self.project_id,
            args.dry_run.unwrap_or(false),
        )
        .await
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        ok_json(&report)
    }

    /// LLM-driven consolidation of a session.
    #[tool(description = "LLM-driven consolidation. Default mode \
        (single-page) rewrites sessions/<id>.md from the observation \
        log. multi_page=true fans out into a batch of concept/decision/\
        gotcha pages plus the session page, all written in one atomic \
        SQL transaction. Off by default; requires \
        AI_MEMORY_LLM_PROVIDER + AI_MEMORY_LLM_MODEL set on the server. \
        Pass dry_run=true to preview without writing.")]
    async fn memory_consolidate(
        &self,
        Parameters(args): Parameters<ConsolidateArgs>,
    ) -> Result<CallToolResult, McpError> {
        let Some(consolidator) = self.consolidator.as_ref() else {
            return Err(McpError::internal_error(
                "memory_consolidate not configured (set AI_MEMORY_LLM_PROVIDER + AI_MEMORY_LLM_MODEL)",
                None,
            ));
        };
        let session_id = SessionId::from_str(&args.session_id)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        let dry = args.dry_run.unwrap_or(false);
        if args.multi_page.unwrap_or(false) {
            let outcomes = consolidator
                .consolidate_session_multi(session_id, dry)
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;
            ok_json(&serde_json::json!({ "outcomes": outcomes }))
        } else {
            let outcome = consolidator
                .consolidate_session(session_id, dry)
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;
            ok_json(&outcome)
        }
    }

    /// Create a handoff snapshot for the next agent CLI.
    #[tool(description = "Record a cross-agent handoff snapshot. Call this \
        before quitting one CLI so the next one (e.g. Codex picking up \
        after Claude Code) can fetch context via memory_handoff_accept. \
        Use cwd to scope the handoff to a specific working directory.")]
    async fn memory_handoff_begin(
        &self,
        Parameters(args): Parameters<HandoffBeginArgs>,
    ) -> Result<CallToolResult, McpError> {
        let handoff = NewHandoff {
            workspace_id: self.workspace_id,
            project_id: self.project_id,
            from_session_id: None,
            from_agent: AgentKind::Other,
            to_agent: None,
            cwd: args.cwd.map(std::path::PathBuf::from),
            summary: args.summary,
            open_questions: args.open_questions,
            next_steps: args.next_steps,
            files_touched: args.files_touched,
        };
        let id = self
            .writer
            .insert_handoff(handoff)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        ok_json(&serde_json::json!({ "handoff_id": id.to_string() }))
    }

    /// Fetch the latest open handoff for this project (optionally filtered
    /// by cwd) and mark it accepted.
    #[tool(description = "Fetch the latest open handoff for this project \
        and mark it accepted. Returns the summary + open questions + next \
        steps so the agent can prepend them to the session context.")]
    async fn memory_handoff_accept(
        &self,
        Parameters(args): Parameters<HandoffAcceptArgs>,
    ) -> Result<CallToolResult, McpError> {
        let handoff = self
            .reader
            .latest_open_handoff(self.workspace_id, self.project_id, args.cwd)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        match handoff {
            None => ok_json(&serde_json::json!({ "handoff": null })),
            Some(h) => {
                self.writer
                    .accept_handoff(h.id, AgentKind::Other, None)
                    .await
                    .map_err(|e| McpError::internal_error(e.to_string(), None))?;
                ok_json(&serde_json::json!({ "handoff": h }))
            }
        }
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

impl AiMemoryServer {
    /// Fire-and-forget access-counter bump for the M8 reinforcement
    /// term. Failures are logged at warn but never surfaced to the
    /// caller.
    fn spawn_access_bump(&self, ids: Vec<PageId>) {
        if ids.is_empty() {
            return;
        }
        let writer = self.writer.clone();
        tokio::spawn(async move {
            if let Err(e) = writer.bump_access(ids).await {
                tracing::warn!(error = %e, "access bump failed");
            }
        });
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

    async fn setup_server() -> (TempDir, Store, AiMemoryServer, WorkspaceId, ProjectId) {
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

        let server = AiMemoryServer::new(store.reader.clone(), store.writer.clone(), ws, proj);
        (tmp, store, server, ws, proj)
    }

    #[tokio::test]
    async fn server_constructs_with_tool_router() {
        let (_tmp, _store, _server, _ws, _pj) = setup_server().await;
    }

    #[tokio::test]
    async fn memory_query_returns_hits_via_tool_method() {
        let (_tmp, _store, server, _ws, _pj) = setup_server().await;
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
        let (_tmp, _store, server, _ws, _pj) = setup_server().await;
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
        let (_tmp, _store, server, _ws, _pj) = setup_server().await;
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

    #[tokio::test]
    async fn handoff_begin_then_accept_round_trips() {
        let (_tmp, _store, server, _ws, _pj) = setup_server().await;
        let begin = server
            .memory_handoff_begin(Parameters(HandoffBeginArgs {
                summary: "left mid-refactor of writer actor".into(),
                open_questions: vec!["what max channel size?".into()],
                next_steps: vec!["finish supersession path".into()],
                files_touched: vec!["crates/ai-memory-store/src/writer.rs".into()],
                cwd: Some("/tmp/aim".into()),
            }))
            .await
            .unwrap();
        let begin_text = begin
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.clone())
            .unwrap();
        assert!(begin_text.contains("handoff_id"));

        // Accepting with matching cwd returns the handoff.
        let accept = server
            .memory_handoff_accept(Parameters(HandoffAcceptArgs {
                cwd: Some("/tmp/aim".into()),
            }))
            .await
            .unwrap();
        let accept_text = accept
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.clone())
            .unwrap();
        assert!(accept_text.contains("left mid-refactor"));
        assert!(accept_text.contains("what max channel size?"));

        // Second accept returns null (handoff is now accepted).
        let again = server
            .memory_handoff_accept(Parameters(HandoffAcceptArgs {
                cwd: Some("/tmp/aim".into()),
            }))
            .await
            .unwrap();
        let again_text = again
            .content
            .first()
            .and_then(|c| c.as_text())
            .map(|t| t.text.clone())
            .unwrap();
        assert!(again_text.contains("\"handoff\": null"));
    }
}
