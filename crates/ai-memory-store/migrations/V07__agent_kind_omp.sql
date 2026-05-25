-- Expand sessions.agent_kind CHECK to include `omp` (Oh My Pi / pi hooks).
--
-- Hook payloads send `agent=pi` (or `omp`), parsed to AgentKind::Omp and
-- persisted as the wire string `omp`. V01 only allowed claude-code/codex/
-- open-code/other, so every pi hook tripped:
--   CHECK constraint failed: agent_kind IN (...)

PRAGMA foreign_keys = OFF;

CREATE TABLE sessions_new (
    id               BLOB PRIMARY KEY NOT NULL,
    workspace_id     BLOB NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    project_id       BLOB NOT NULL REFERENCES projects(id)   ON DELETE CASCADE,
    agent_kind       TEXT NOT NULL CHECK (agent_kind IN ('claude-code','codex','open-code','omp','other')),
    cwd              TEXT,
    started_at       INTEGER NOT NULL,
    ended_at         INTEGER,
    summary_page_id  BLOB REFERENCES pages(id) ON DELETE SET NULL
);

INSERT INTO sessions_new SELECT * FROM sessions;

DROP TABLE sessions;

ALTER TABLE sessions_new RENAME TO sessions;

CREATE INDEX idx_sessions_recent ON sessions(workspace_id, project_id, started_at DESC);
CREATE INDEX idx_sessions_project ON sessions(project_id);

PRAGMA foreign_keys = ON;
