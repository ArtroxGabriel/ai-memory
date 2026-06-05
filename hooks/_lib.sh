# ai-memory hook helper — find marker file + parse minimal TOML.
# Sourced by per-agent lifecycle hook scripts. POSIX shell only —
# no bash-isms, no non-standard deps (no jq, no toml crate). Keep changes
# byte-trivial because every supported agent (claude-code, codex,
# cursor, gemini-cli, antigravity-cli, opencode, omp) sources this same file.

# Walk up from "$1" toward $HOME (or /) looking for `.ai-memory.toml`.
# Prints the absolute path of the first marker found, or nothing.
# Stops at $HOME to avoid leaking declarations from a shared system
# user's home into another user's session on multi-user boxes.
ai_memory_find_marker() {
    dir="$1"
    [ -z "$dir" ] && return 0
    while [ -n "$dir" ] && [ "$dir" != "/" ]; do
        if [ -f "$dir/.ai-memory.toml" ]; then
            printf '%s\n' "$dir/.ai-memory.toml"
            return 0
        fi
        if [ -n "${HOME:-}" ] && [ "$dir" = "$HOME" ]; then
            return 0
        fi
        parent=$(dirname "$dir")
        [ "$parent" = "$dir" ] && return 0
        dir="$parent"
    done
}

# Parse `key = "value"` at the TOML root (no nesting, no arrays, no
# tables). Returns the first match or nothing. Ignores comments and
# blank lines by construction (the regex only matches the `key = "..."`
# shape).
ai_memory_parse_toml_key() {
    file="$1"; key="$2"
    [ -f "$file" ] || return 0
    sed -n -E "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*\"([^\"]*)\".*/\1/p" \
        "$file" | head -n 1
}

# Extract the first cwd-like path from a JSON payload on stdin or in $1.
# Returns the value or nothing. This is intentionally a tiny shell fallback,
# not a JSON parser; taking the first match preserves the top-level cwd when
# tool payloads contain nested `cwd` fields later in the object. Antigravity
# CLI sends `workspacePaths: ["/repo", ...]` instead of `cwd`.
ai_memory_extract_cwd() {
    payload="${1:-$(cat)}"
    rest=${payload#*\"cwd\"}
    if [ "$rest" != "$payload" ]; then
        printf '%s' "$rest" \
            | sed -n -E 's/^[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/p' \
            | head -n 1
        return 0
    fi
    rest=${payload#*\"workspacePaths\"}
    [ "$rest" = "$payload" ] && return 0
    printf '%s' "$rest" \
        | sed -n -E 's/^[[:space:]]*:[[:space:]]*\[[[:space:]]*"([^"]*)".*/\1/p' \
        | head -n 1
}

# URL-encode the minimal set of characters that have meaning in a query
# string. Sufficient for the schema's value regex (`^[a-z0-9][a-z0-9._-]*$`)
# plus a defensive pass for anything a hand-edited marker might contain.
ai_memory_url_encode() {
    printf '%s' "$1" \
        | sed 's/%/%25/g; s/+/%2B/g; s/&/%26/g; s/=/%3D/g; s/?/%3F/g; s/#/%23/g; s/ /%20/g'
}

# Build a query-string suffix from "$1" plus any marker file walked up from
# it. Returns the suffix with the leading `&`, or nothing when cwd is absent.
# `cwd` is always included so `GET /handoff` resolves the same basename project
# as the prior hook events even when no marker file exists.
ai_memory_marker_qs() {
    cwd="$1"
    [ -z "$cwd" ] && return 0
    qs="&cwd=$(ai_memory_url_encode "$cwd")"
    marker=$(ai_memory_find_marker "$cwd")
    [ -z "$marker" ] && { printf '%s' "$qs"; return 0; }
    ws=$(ai_memory_parse_toml_key "$marker" workspace)
    pr=$(ai_memory_parse_toml_key "$marker" project)
    st=$(ai_memory_parse_toml_key "$marker" project_strategy)
    [ -n "$ws" ] && qs="${qs}&workspace=$(ai_memory_url_encode "$ws")"
    [ -n "$pr" ] && qs="${qs}&project=$(ai_memory_url_encode "$pr")"
    [ -n "$st" ] && qs="${qs}&project_strategy=$(ai_memory_url_encode "$st")"
    printf '%s' "$qs"
}

# POST stdin to "$1" as JSON, fire-and-forget. Adds an
# `Authorization: Bearer` header when `AI_MEMORY_AUTH_TOKEN` is set.
# The 0.5s timeout matches the project-wide hook latency budget
# (never block the agent), and the trailing `|| true` makes the
# function safe to call from `set -e` scripts.
ai_memory_post_hook() {
    if [ -n "${AI_MEMORY_AUTH_TOKEN:-}" ]; then
        curl -s --max-time 0.5 -X POST "$1" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $AI_MEMORY_AUTH_TOKEN" \
            --data-binary @-
    else
        curl -s --max-time 0.5 -X POST "$1" \
            -H "Content-Type: application/json" \
            --data-binary @-
    fi
}

# GET "$1" with the same auth-header rules as `ai_memory_post_hook`.
# Used by `session-start.sh` to pull the cross-agent handoff before
# the resuming agent's first prompt. 1s budget — slightly more
# generous than POST because the result is *synchronously* fed to
# stdout (and prepended to the agent's context), so we want to avoid
# truncating a handoff that was almost ready.
ai_memory_get_handoff() {
    if [ -n "${AI_MEMORY_AUTH_TOKEN:-}" ]; then
        curl -s --max-time 1.0 "$1" \
            -H "Authorization: Bearer $AI_MEMORY_AUTH_TOKEN"
    else
        curl -s --max-time 1.0 "$1"
    fi
}

# Encode stdin as a JSON string (with surrounding quotes). Used by hooks
# whose stdout contract is JSON rather than raw context text: Antigravity's
# PreInvocation hook and Claude Code's session-start hook (which wraps the
# handoff in hookSpecificOutput.additionalContext).
ai_memory_json_string() {
    awk '
        BEGIN { printf "\"" }
        {
            gsub(/\\/, "\\\\")
            gsub(/"/, "\\\"")
            gsub(/\t/, "\\t")
            gsub(/\r/, "\\r")
            printf "%s%s", sep, $0
            sep = "\\n"
        }
        END { printf "\"" }
    '
}
