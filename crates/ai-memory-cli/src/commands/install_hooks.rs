//! `ai-memory install-hooks` — print the suggested lifecycle-hook
//! configuration for the chosen agent CLI.
//!
//! In M3 this is *non-destructive*: we render the JSON snippet the user
//! should merge into their agent CLI's settings file, plus the absolute
//! paths to the vendored shell scripts. We intentionally do not mutate
//! `~/.claude/settings.json` automatically — agent CLI hook formats are
//! still in flux and bad merges are very user-visible.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::json;

use crate::cli::{AgentChoice, InstallHooksArgs};
use crate::config::Config;

/// Run the `install-hooks` subcommand.
///
/// # Errors
/// Returns an error if the hook script directory cannot be located.
pub fn run(_config: &Config, args: InstallHooksArgs) -> Result<()> {
    let hooks_dir = resolve_hooks_dir(args.hooks_dir.as_deref(), args.agent)?;
    match args.agent {
        AgentChoice::ClaudeCode => render_claude_code(&hooks_dir, &args.server_url),
        AgentChoice::Codex => render_agent("codex", &hooks_dir, &args.server_url),
        AgentChoice::OpenCode => render_agent("opencode", &hooks_dir, &args.server_url),
    }
}

fn render_agent(label: &str, hooks_dir: &Path, server_url: &str) -> Result<()> {
    println!("# {label} hook scripts (manual install — wire each to the matching event)");
    println!("# Hook scripts: {}", hooks_dir.display());
    println!("# AI-memory server URL: {server_url}");
    println!();
    for entry in std::fs::read_dir(hooks_dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() && p.extension().is_some_and(|e| e == "sh") {
            println!("- {}", p.display());
        }
    }
    println!();
    println!("Set AI_MEMORY_HOOK_URL in each hook's environment to override the default.");
    Ok(())
}

fn resolve_hooks_dir(explicit: Option<&Path>, agent: AgentChoice) -> Result<PathBuf> {
    let sub = match agent {
        AgentChoice::ClaudeCode => "claude-code",
        AgentChoice::Codex => "codex",
        AgentChoice::OpenCode => "opencode",
    };
    if let Some(p) = explicit {
        let path = p.join(sub);
        if path.is_dir() {
            return Ok(path);
        }
        anyhow::bail!("hooks directory {} does not exist", path.display());
    }

    // Probe candidates in order. The first dir that exists wins.
    let candidates: [PathBuf; 3] = [
        // Cargo-run from the repo.
        repo_root_guess()
            .map(|r| r.join("hooks").join(sub))
            .unwrap_or_default(),
        // Docker image lays them out under /usr/local/share/ai-memory/.
        PathBuf::from(format!("/usr/local/share/ai-memory/hooks/{sub}")),
        // Local install honourable mention.
        dirs::data_local_dir()
            .map(|d| d.join("ai-memory/hooks").join(sub))
            .unwrap_or_default(),
    ];
    for path in &candidates {
        if !path.as_os_str().is_empty() && path.is_dir() {
            return Ok(path.clone());
        }
    }
    anyhow::bail!("could not locate hooks directory. Tried: {:?}", candidates,);
}

fn repo_root_guess() -> Option<PathBuf> {
    // When the binary lives under target/{debug,release}/<name>, the
    // workspace root is two parents up.
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent()?.parent()?.parent().map(Path::to_path_buf))
}

fn render_claude_code(hooks_dir: &Path, server_url: &str) -> Result<()> {
    let scripts: [(&str, &str); 7] = [
        ("SessionStart", "session-start.sh"),
        ("UserPromptSubmit", "user-prompt-submit.sh"),
        ("PreToolUse", "pre-tool-use.sh"),
        ("PostToolUse", "post-tool-use.sh"),
        ("PreCompact", "pre-compact.sh"),
        ("Stop", "stop.sh"),
        ("SessionEnd", "session-end.sh"),
    ];

    let mut hooks_block = serde_json::Map::new();
    for (event, script) in scripts {
        let abs = hooks_dir.join(script);
        if !abs.exists() {
            anyhow::bail!("missing hook script: {}", abs.display());
        }
        hooks_block.insert(
            event.into(),
            json!([{
                "command": abs.to_string_lossy().into_owned(),
                "env": { "AI_MEMORY_HOOK_URL": server_url },
            }]),
        );
    }
    let payload = json!({ "hooks": hooks_block });

    let serialized =
        serde_json::to_string_pretty(&payload).context("serializing claude code hook config")?;
    println!("# Claude Code hook config — merge into ~/.claude/settings.json");
    println!("# Hook scripts: {}", hooks_dir.display());
    println!("# AI-memory server URL: {server_url}");
    println!();
    println!("{serialized}");
    Ok(())
}
