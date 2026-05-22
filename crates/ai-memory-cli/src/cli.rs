//! Command-line interface definition (clap derive).

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// Top-level CLI for the `ai-memory` binary.
#[derive(Debug, Parser)]
#[command(name = "ai-memory", version, about, long_about = None)]
pub struct Cli {
    /// Override the data directory.
    ///
    /// Defaults to a platform path under `dirs::data_local_dir()`. Also
    /// settable via the `AI_MEMORY_DATA_DIR` environment variable.
    #[arg(long, env = "AI_MEMORY_DATA_DIR", global = true)]
    pub data_dir: Option<PathBuf>,

    /// Path to an explicit config file (defaults to `<data_dir>/config.toml`).
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Initialise the data directory layout.
    Init(InitArgs),
    /// Print runtime status (counts, paths, version).
    Status(StatusArgs),
    /// Full-text search the wiki via FTS5.
    Search(SearchArgs),
    /// Write or update a wiki page atomically (also indexes it in the store).
    WritePage(WritePageArgs),
    /// Run the filesystem watcher (foreground; Ctrl-C to stop).
    Watch(WatchArgs),
    /// Run the MCP server (with watcher) over stdio or HTTP.
    Serve(ServeArgs),
    /// Wipe the data directory's wiki/, db/, raw/ contents.
    Reset(ResetArgs),
    /// Snapshot wiki/, db/, and config.toml into a gzipped tarball.
    Backup(BackupArgs),
    /// Restore a backup tarball into the data directory.
    Restore(RestoreArgs),
    /// Print (or apply) lifecycle-hook configuration for an agent CLI.
    InstallHooks(InstallHooksArgs),
    /// Stage + commit the wiki tree under git.
    Commit(CommitArgs),
    /// Smoke-test an LLM provider by sending one prompt.
    LlmTest(LlmTestArgs),
    /// Run the M8 retention sweep over episodic pages.
    ForgetSweep(ForgetSweepArgs),
    /// Run the M8 lint pass (stale / duplicates + optional LLM contradiction).
    Lint(LintArgs),
    /// Compute + store embeddings for every latest page (M9).
    Embed(EmbedArgs),
}

/// Arguments for `init`.
#[derive(Debug, Args)]
pub struct InitArgs {
    /// Overwrite an existing `config.toml` if present.
    #[arg(long)]
    pub force: bool,
}

/// Arguments for `status`.
#[derive(Debug, Args)]
pub struct StatusArgs {
    /// Emit the report as JSON instead of human-readable text.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `search`.
#[derive(Debug, Args)]
pub struct SearchArgs {
    /// FTS5 query string (e.g. `"karpathy wiki"` or `quick OR slow`).
    pub query: String,
    /// Maximum number of hits to return.
    #[arg(short = 'n', long, default_value_t = 10)]
    pub limit: usize,
    /// Emit results as JSON.
    #[arg(long)]
    pub json: bool,
}

/// Arguments for `watch`.
#[derive(Debug, Args)]
pub struct WatchArgs {
    /// Workspace name to attribute discovered files to (auto-created).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}

/// Arguments for `reset`.
#[derive(Debug, Args)]
pub struct ResetArgs {
    /// Required to actually wipe data. Without this we just dry-run.
    #[arg(long)]
    pub confirm: bool,
}

/// Arguments for `backup`.
#[derive(Debug, Args)]
pub struct BackupArgs {
    /// Destination tarball (`.tar.gz`).
    #[arg(long, short = 'o')]
    pub to: PathBuf,
}

/// Arguments for `restore`.
#[derive(Debug, Args)]
pub struct RestoreArgs {
    /// Source tarball.
    #[arg(long, short = 'i')]
    pub from: PathBuf,
    /// Overwrite an existing non-empty data dir.
    #[arg(long)]
    pub force: bool,
}

/// Agent CLI to install hooks for.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum AgentChoice {
    /// Anthropic Claude Code.
    ClaudeCode,
    /// OpenAI Codex CLI.
    Codex,
    /// OpenCode (open-source coding agent).
    OpenCode,
}

/// Arguments for `commit`.
#[derive(Debug, Args)]
pub struct CommitArgs {
    /// Commit message.
    #[arg(long, short = 'm', default_value = "manual commit")]
    pub message: String,
}

/// LLM provider for `llm-test`.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum LlmProviderChoice {
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI Chat Completions.
    Openai,
    /// OpenAI-compatible local (Ollama, vLLM, LM Studio).
    OpenaiCompat,
}

/// Arguments for `embed`.
#[derive(Debug, Args)]
pub struct EmbedArgs {
    /// Report what would be embedded without actually mutating.
    #[arg(long)]
    pub dry_run: bool,
    /// Re-embed pages even when they already have a row with the
    /// currently-configured `(provider, model, dim)`.
    #[arg(long)]
    pub force: bool,
    /// Workspace name (auto-created if absent).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created if absent).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}

/// Arguments for `forget-sweep`.
#[derive(Debug, Args)]
pub struct ForgetSweepArgs {
    /// Report what would be evicted without actually mutating.
    #[arg(long)]
    pub dry_run: bool,
}

/// Arguments for `lint`.
#[derive(Debug, Args)]
pub struct LintArgs {
    /// Compute findings but don't write `wiki/_lint/<date>.md`.
    #[arg(long)]
    pub dry_run: bool,
    /// Workspace name (auto-created if absent).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created if absent).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}

/// Arguments for `llm-test`.
#[derive(Debug, Args)]
pub struct LlmTestArgs {
    /// Provider to test.
    #[arg(long, value_enum)]
    pub provider: LlmProviderChoice,
    /// Model identifier (e.g. `claude-sonnet-4-7`, `gpt-4o-mini`, `llama3.1:8b`).
    #[arg(long)]
    pub model: String,
    /// Prompt to send.
    #[arg(long)]
    pub prompt: String,
    /// Base URL override (required for openai-compat).
    #[arg(long, env = "LLM_BASE_URL")]
    pub base_url: Option<String>,
    /// Optional API key override (otherwise pulled from env).
    #[arg(long, env = "LLM_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,
}

/// Arguments for `install-hooks`.
#[derive(Debug, Args)]
pub struct InstallHooksArgs {
    /// Which agent's hooks to render.
    #[arg(long, value_enum, default_value_t = AgentChoice::ClaudeCode)]
    pub agent: AgentChoice,
    /// Filesystem root that contains the vendored hook scripts (defaults
    /// to the repo's `hooks/` if known, else `/usr/local/share/ai-memory/hooks`).
    #[arg(long)]
    pub hooks_dir: Option<PathBuf>,
    /// Server URL the hooks will POST to.
    #[arg(long, default_value = "http://127.0.0.1:7777")]
    pub server_url: String,
}

/// Transport for the MCP server.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum TransportKind {
    /// Stdio — what `claude mcp add` uses.
    Stdio,
    /// Streamable HTTP — for HTTP clients and `mcp-inspector`.
    Http,
}

/// Arguments for `serve`.
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Transport to expose the MCP server on.
    #[arg(long, value_enum, default_value_t = TransportKind::Stdio)]
    pub transport: TransportKind,
    /// Bind address for `--transport http` (default: from config).
    #[arg(long)]
    pub bind: Option<String>,
    /// Skip the filesystem watcher; useful for transient debugging.
    #[arg(long)]
    pub no_watcher: bool,
    /// Workspace name (auto-created).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}

/// Arguments for `write-page`.
#[derive(Debug, Args)]
pub struct WritePageArgs {
    /// Relative wiki path (e.g. `notes/foo.md`).
    #[arg(long, visible_alias = "p")]
    pub path: String,
    /// Markdown body. Use `-` to read from stdin.
    #[arg(long, visible_alias = "b")]
    pub body: String,
    /// Optional page title; otherwise derived from the first `# heading`
    /// in the body, or the path stem.
    #[arg(long)]
    pub title: Option<String>,
    /// Repeatable tag to add to the frontmatter `tags` array.
    #[arg(long, short = 't')]
    pub tag: Vec<String>,
    /// Tier (`working`, `episodic`, `semantic`, `procedural`).
    #[arg(long, default_value = "semantic")]
    pub tier: String,
    /// Pin the page so the future decay sweep skips it.
    #[arg(long)]
    pub pinned: bool,
    /// Workspace name (auto-created if absent).
    #[arg(long, default_value = "default")]
    pub workspace: String,
    /// Project name within the workspace (auto-created if absent).
    #[arg(long, default_value = "scratch")]
    pub project: String,
}
