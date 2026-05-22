//! axum router exposing `POST /hook`.
//!
//! Always returns 202 immediately. Heavy work (DB writes, session-page
//! synthesis) happens *after* the response is sent — but we still
//! `await` the writer ack to honour the cross-cutting invariant that
//! "indexes commit in the same transaction as the data" (no
//! background-task-indexing-after-return, basic-memory #763). The agent
//! never blocks on us thanks to the fire-and-forget client side.

use std::str::FromStr;
use std::sync::Arc;

use ai_memory_core::{
    AgentKind, NewHandoff, NewObservation, NewSession, ObservationKind, ProjectId, SessionId,
    WorkspaceId,
};
use ai_memory_store::WriterHandle;
use ai_memory_wiki::Wiki;
use axum::Json;
use axum::Router;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::post;
use jiff::Timestamp;
use tracing::{info, warn};
use uuid::Uuid;

use crate::log;
use crate::payload::{HookEnvelope, HookEvent, HookQuery};
use crate::sanitize::Sanitized;
use crate::synth::synthesize_session_page;

/// Shared state passed to the hook handler.
#[derive(Clone)]
pub struct HookState {
    /// Workspace these observations belong to.
    pub workspace_id: WorkspaceId,
    /// Project these observations belong to.
    pub project_id: ProjectId,
    /// Writer actor handle.
    pub writer: WriterHandle,
    /// Reader pool — needed for session-end synthesis.
    pub reader: ai_memory_store::ReaderPool,
    /// Wiki handle — used to write the session-summary page.
    pub wiki: Wiki,
}

/// Build a router with the single `POST /hook` route mounted on the
/// returned [`Router`].
pub fn hook_router(state: HookState) -> Router {
    Router::new()
        .route("/hook", post(handle_hook))
        .with_state(Arc::new(state))
}

async fn handle_hook(
    State(state): State<Arc<HookState>>,
    Query(query): Query<HookQuery>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let env = HookEnvelope::from_query_and_body(query, body);
    tokio::spawn(process_envelope(state.clone(), env));
    (StatusCode::ACCEPTED, "queued")
}

async fn process_envelope(state: Arc<HookState>, env: HookEnvelope) {
    if let Err(e) = process(&state, env).await {
        warn!(error = %e, "hook processing failed");
    }
}

async fn process(state: &HookState, env: HookEnvelope) -> anyhow::Result<()> {
    let session_id = resolve_session_id(&env)?;

    // Begin the session row if SessionStart, otherwise no-op (the
    // `INSERT OR IGNORE` makes this safe).
    if matches!(env.event, HookEvent::SessionStart) {
        let new_session = NewSession {
            id: session_id,
            workspace_id: state.workspace_id,
            project_id: state.project_id,
            agent_kind: env.agent,
            cwd: env.cwd.as_ref().map(std::path::PathBuf::from),
        };
        state.writer.begin_session(new_session).await?;
    }

    // Persist the observation row.
    let kind = env.event.to_observation_kind();
    let title = env
        .title_hint
        .clone()
        .unwrap_or_else(|| kind.as_str().to_string());
    let body = env.body_excerpt.clone().unwrap_or_default();
    let raw_obs = NewObservation {
        session_id,
        workspace_id: state.workspace_id,
        project_id: state.project_id,
        kind,
        title,
        body,
        importance: importance_for(env.event),
    };
    let sanitized = Sanitized::new(raw_obs);
    let _ = state
        .writer
        .insert_observation(sanitized.inner().clone())
        .await?;

    // Append the log line.
    if let Err(e) = log::append_event(
        state.wiki.root(),
        Timestamp::now(),
        env.event,
        sanitized.inner().title.as_str(),
    ) {
        warn!(error = %e, "log.md append failed");
    }

    // On SessionEnd, synthesize the summary page, end the session, and
    // auto-create a handoff so the next agent can pick up.
    if matches!(env.event, HookEvent::SessionEnd) {
        let observations = state.reader.observations_for_session(session_id).await?;
        let new_page = synthesize_session_page(
            state.workspace_id,
            state.project_id,
            session_id,
            &observations,
        );
        let page_id = state
            .wiki
            .write_page(ai_memory_wiki::WritePageRequest {
                workspace_id: new_page.workspace_id,
                project_id: new_page.project_id,
                path: new_page.path.clone(),
                frontmatter: new_page.frontmatter_json.clone(),
                body: new_page.body.clone(),
                tier: new_page.tier,
                pinned: new_page.pinned,
            })
            .await?;
        state.writer.end_session(session_id, Some(page_id)).await?;
        let handoff =
            build_auto_handoff(state, env.agent, session_id, env.cwd.clone(), &observations);
        let handoff_id = state.writer.insert_handoff(handoff).await?;
        info!(
            session = %session_id,
            page = %new_page.path,
            handoff = %handoff_id,
            "session ended; summary page + open handoff created",
        );
    }

    Ok(())
}

fn resolve_session_id(env: &HookEnvelope) -> anyhow::Result<SessionId> {
    if let Some(raw) = &env.session_id {
        // Accept either a UUID (canonical) or any string, hashing the
        // latter to a deterministic UUID v5 so each agent's session id
        // maps cleanly into our schema.
        if let Ok(id) = SessionId::from_str(raw) {
            return Ok(id);
        }
        let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, raw.as_bytes());
        return Ok(SessionId(uuid));
    }
    if matches!(env.event, HookEvent::SessionStart) {
        return Ok(SessionId::new());
    }
    anyhow::bail!("hook payload missing session_id and event is not session-start")
}

fn build_auto_handoff(
    state: &HookState,
    from_agent: AgentKind,
    session_id: SessionId,
    cwd: Option<String>,
    observations: &[ai_memory_core::Observation],
) -> NewHandoff {
    let mut prompts: Vec<&str> = Vec::new();
    let mut tools: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for obs in observations {
        match obs.kind {
            ObservationKind::UserPrompt if !obs.title.is_empty() => prompts.push(&obs.title),
            ObservationKind::PostToolUse | ObservationKind::PreToolUse if !obs.title.is_empty() => {
                tools.insert(&obs.title);
            }
            _ => {}
        }
    }
    let last_prompt = prompts.last().copied();
    let summary = match (prompts.first().copied(), last_prompt) {
        (Some(first), Some(last)) if first == last => format!("Session focused on: {first}"),
        (Some(first), Some(last)) => format!("Started: {first}. Last: {last}."),
        (Some(first), None) => format!("Started: {first}."),
        _ => format!(
            "Session ended; {} observations recorded.",
            observations.len()
        ),
    };
    let open_questions = if prompts.is_empty() {
        Vec::new()
    } else {
        // Heuristic: last user prompt often *is* the open question.
        vec![format!(
            "Continue from: {}",
            last_prompt.unwrap_or_default()
        )]
    };
    let next_steps = if tools.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "Tools used: {}",
            tools.into_iter().collect::<Vec<_>>().join(", ")
        )]
    };
    NewHandoff {
        workspace_id: state.workspace_id,
        project_id: state.project_id,
        from_session_id: Some(session_id),
        from_agent,
        to_agent: None,
        cwd: cwd.map(std::path::PathBuf::from),
        summary,
        open_questions,
        next_steps,
        files_touched: Vec::new(),
    }
}

const fn importance_for(event: HookEvent) -> u8 {
    match event {
        HookEvent::SessionStart | HookEvent::SessionEnd => 7,
        HookEvent::UserPrompt => 8,
        HookEvent::PostToolUse | HookEvent::PreToolUse => 5,
        HookEvent::Stop | HookEvent::PreCompact => 6,
        HookEvent::Notification | HookEvent::Other => 3,
    }
}

#[allow(dead_code)]
fn _agent_kind_marker(_k: AgentKind) {}
