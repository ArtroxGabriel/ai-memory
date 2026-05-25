//! FTS5 `MATCH` query preparation for user/agent-supplied search text.
//!
//! FTS5 treats `column:term` as a column-qualified search. Natural-language
//! queries that contain colons (`pick: handoff`, `memory: bootstrap`) make
//! SQLite error with `no such column: pick` because only `title` and `body`
//! exist on `pages_fts`. Colons are neutralised before each token is quoted.

/// Sanitize free-text for use in `WHERE pages_fts MATCH ?`.
///
/// Returns an empty string when `raw` is empty/whitespace-only; callers
/// should skip the SQL query in that case.
#[must_use]
pub fn prepare_fts5_query(raw: &str) -> String {
    // Turn `pick:` / `memory:` into separate tokens so we never emit FTS5
    // column syntax, while still matching indexed words (`pick`, `memory`).
    let normalized: String = raw
        .chars()
        .map(|c| if c == ':' { ' ' } else { c })
        .collect();
    let tokens: Vec<String> = normalized
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .map(quote_fts5_token)
        .collect();
    tokens.join(" ")
}

fn quote_fts5_token(token: &str) -> String {
    // FTS5 escapes `"` inside a quoted token by doubling it.
    let escaped = token.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colon_is_not_column_syntax() {
        let q = prepare_fts5_query("pick: handoff ai-memory");
        assert_eq!(q, "\"pick\" \"handoff\" \"ai-memory\"");
    }

    #[test]
    fn empty_yields_empty() {
        assert_eq!(prepare_fts5_query("   "), "");
    }

    #[test]
    fn quotes_are_escaped() {
        let q = prepare_fts5_query(r#"say "hello""#);
        assert_eq!(q, r#""say" """hello""""#);
    }
}
