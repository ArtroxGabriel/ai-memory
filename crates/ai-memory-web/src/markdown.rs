//! Markdown → HTML rendering with `syntect` syntax highlighting.
//!
//! Stub at scaffold time; full implementation in the next step.

use pulldown_cmark::{CowStr, Event, Options, Parser, Tag, html};

/// Render a markdown body to HTML using GFM-ish defaults.
///
/// Raw HTML is escaped and unsafe link/image schemes are neutralised.
/// Wiki content can be derived from prompts, hooks, or LLM output, so
/// the browser surface must treat it as untrusted.
///
/// `[[wiki links]]` (with the `[[target|label]]`, `[[project:path]]`, and
/// `[[workspace/project:path]]` variants the engine's link extractor
/// understands) are rendered as clickable internal links resolved against
/// the page's own `workspace`/`project` unless the target carries its own
/// scope. Wikilinks inside fenced code blocks and inline code are left as
/// literal text.
#[must_use]
pub fn render(body: &str, workspace: &str, project: &str) -> String {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_FOOTNOTES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);
    opts.insert(Options::ENABLE_SMART_PUNCTUATION);

    // Rewrite `[[wikilinks]]` into ordinary markdown links BEFORE parsing.
    // pulldown-cmark consumes `[...]` as reference-link syntax, so the brackets
    // never survive as a single text node — preprocessing the source is the
    // robust hook. Code (fenced + inline) is skipped so `[[…]]` stays literal
    // there, mirroring the engine's link extractor.
    let body = preprocess_wikilinks(body, workspace, project);

    let parser = Parser::new_ext(&body, opts).map(sanitize_event);
    let mut out = String::with_capacity(body.len() + body.len() / 4);
    html::push_html(&mut out, parser);
    out
}

/// Convert `[[target]]` / `[[target|label]]` spans into `[label](href)`
/// markdown links, skipping fenced code blocks and inline-code spans. Targets
/// that aren't internal pages (external schemes, traversal, empty) are left as
/// literal `[[…]]`.
fn preprocess_wikilinks(body: &str, workspace: &str, project: &str) -> String {
    let mut out = String::with_capacity(body.len() + 64);
    let mut in_fence = false;
    for line in body.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            out.push_str(line);
            continue;
        }
        if in_fence {
            out.push_str(line);
            continue;
        }
        // Split on backticks: even segments are outside inline code, odd ones
        // are inside it (left verbatim). Unbalanced backticks degrade safely.
        for (i, seg) in line.split('`').enumerate() {
            if i > 0 {
                out.push('`');
            }
            if i % 2 == 0 {
                rewrite_wikilinks_in_text(seg, workspace, project, &mut out);
            } else {
                out.push_str(seg);
            }
        }
    }
    out
}

/// Rewrite every `[[…]]` in a non-code text run into a markdown link.
fn rewrite_wikilinks_in_text(seg: &str, workspace: &str, project: &str, out: &mut String) {
    let mut rest = seg;
    loop {
        let Some(open) = rest.find("[[") else {
            out.push_str(rest);
            break;
        };
        let after = &rest[open + 2..];
        let Some(close) = after.find("]]") else {
            out.push_str(rest); // unterminated → literal
            break;
        };
        out.push_str(&rest[..open]);
        let raw = &after[..close];
        match wikilink_href_label(raw, workspace, project) {
            Some((href, label)) => {
                out.push('[');
                out.push_str(&escape_link_label(&label));
                out.push_str("](");
                out.push_str(&href);
                out.push(')');
            }
            None => {
                out.push_str("[[");
                out.push_str(raw);
                out.push_str("]]");
            }
        }
        rest = &after[close + 2..];
    }
}

/// Escape the characters that would prematurely close a markdown link label.
fn escape_link_label(label: &str) -> String {
    label
        .replace('\\', r"\\")
        .replace('[', r"\[")
        .replace(']', r"\]")
}

/// Resolve a `[[…]]` target (inner text, no brackets) into a relative page
/// href + display label. Returns `None` for empty/external/malformed targets
/// (callers then keep the literal `[[…]]`).
fn wikilink_href_label(raw: &str, workspace: &str, project: &str) -> Option<(String, String)> {
    let (target, label) = match raw.split_once('|') {
        Some((t, l)) => (t.trim(), Some(l.trim())),
        None => (raw.trim(), None),
    };
    if target.is_empty() {
        return None;
    }
    // External / scheme-qualified targets are not internal wiki pages.
    let lower = target.to_ascii_lowercase();
    if target.contains("://")
        || lower.starts_with("mailto:")
        || lower.starts_with("data:")
        || lower.starts_with("javascript:")
        || lower.starts_with("tel:")
        || target.starts_with('#')
    {
        return None;
    }

    // Optional `[workspace/]project:` scope qualifier.
    let (ws, proj, path_part) = split_scope(target, workspace, project);

    // Strip anchor/query, reject traversal, normalise the `.md` suffix.
    let path = path_part.split(['#', '?']).next().unwrap_or("").trim();
    if path.is_empty() || path.contains('\\') || path.contains("..") {
        return None;
    }
    let path = if path.rsplit('/').next().is_some_and(|seg| seg.contains('.')) {
        path.to_string()
    } else {
        format!("{path}.md")
    };

    let href = crate::templates::page_href(ws, proj, &path);
    let display = label
        .filter(|l| !l.is_empty())
        .unwrap_or(target)
        .to_string();
    Some((href, display))
}

/// Peel an optional `[workspace/]project:` scope off a wikilink target,
/// defaulting to the current page's `workspace`/`project`. A `scope` made of
/// anything other than `[-_/.alnum]` is treated as part of the path (so a
/// stray colon in a filename doesn't masquerade as a scope).
fn split_scope<'a>(
    target: &'a str,
    cur_ws: &'a str,
    cur_proj: &'a str,
) -> (&'a str, &'a str, &'a str) {
    if let Some((scope, rest)) = target.split_once(':') {
        let scope = scope.trim();
        let scope_ok = !scope.is_empty()
            && scope
                .chars()
                .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '/' | '.'));
        if scope_ok {
            return match scope.split_once('/') {
                Some((ws, proj)) if !proj.trim().is_empty() => {
                    (ws.trim(), proj.trim(), rest.trim())
                }
                Some(_) => (cur_ws, cur_proj, target), // malformed `ws/`
                None => (cur_ws, scope, rest.trim()),
            };
        }
    }
    (cur_ws, cur_proj, target)
}

fn sanitize_event(event: Event<'_>) -> Event<'_> {
    match event {
        Event::Html(s) | Event::InlineHtml(s) => Event::Text(s),
        Event::Start(Tag::Link {
            link_type,
            dest_url,
            title,
            id,
        }) => Event::Start(Tag::Link {
            link_type,
            dest_url: safe_url(dest_url),
            title,
            id,
        }),
        Event::Start(Tag::Image {
            link_type,
            dest_url,
            title,
            id,
        }) => Event::Start(Tag::Image {
            link_type,
            dest_url: safe_url(dest_url),
            title,
            id,
        }),
        other => other,
    }
}

fn safe_url(url: CowStr<'_>) -> CowStr<'_> {
    let trimmed = url.trim_start();
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("mailto:")
        || lower.starts_with('/')
        || lower.starts_with('#')
        || !lower.contains(':')
    {
        url
    } else {
        CowStr::Boxed("#".into())
    }
}

/// Escape text for insertion into an HTML template while preserving the
/// fixed `<mark>` tags emitted by SQLite FTS snippets.
#[must_use]
pub fn escape_snippet(snippet: &str) -> String {
    escape_html(snippet)
        .replace("&lt;mark&gt;", "<mark>")
        .replace("&lt;/mark&gt;", "</mark>")
}

fn escape_html(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for c in input.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(c),
        }
    }
    out
}

/// Drop the leading H1 from a markdown body if present. Static-site
/// convention: the first H1 IS the page title, and the page template
/// already renders the title in its header — leaving it in the body
/// duplicates it on screen. No-op when the body doesn't start with
/// an H1 (handles `# Title`, both ATX `# Title` and setext
/// `Title\n=====` forms).
#[must_use]
pub fn strip_leading_h1(body: &str) -> &str {
    // Skip any leading blank lines.
    let trimmed = body.trim_start_matches(['\n', '\r']);
    // ATX form: `# Title` (one `#`, NOT `## …`).
    if let Some(rest) = trimmed.strip_prefix("# ") {
        let after_line = rest.find('\n').map_or("", |nl| &rest[nl + 1..]);
        return after_line.trim_start_matches(['\n', '\r']);
    }
    // Setext form: `Title\n====…` (1+ equals signs). Look ahead.
    if let Some((first_line, after_first)) = trimmed.split_once('\n')
        && !first_line.is_empty()
        && let Some((second_line, after_second)) = after_first.split_once('\n')
        && !second_line.is_empty()
        && second_line.chars().all(|c| c == '=')
    {
        return after_second.trim_start_matches(['\n', '\r']);
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_basic_markdown() {
        let html = render("# Hello\n\nworld", "default", "scratch");
        assert!(html.contains("<h1>Hello</h1>"));
        assert!(html.contains("<p>world</p>"));
    }

    #[test]
    fn renders_tables() {
        let md = "| a | b |\n|---|---|\n| 1 | 2 |";
        let html = render(md, "default", "scratch");
        assert!(html.contains("<table>"));
        assert!(html.contains("<td>1</td>"));
    }

    #[test]
    fn escapes_raw_html() {
        let html = render("<script>alert(1)</script>", "default", "scratch");
        assert!(html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
        assert!(!html.contains("<script>"));
    }

    #[test]
    fn neutralises_javascript_links() {
        let html = render("[x](javascript:alert(1))", "default", "scratch");
        assert!(html.contains("href=\"#\""));
        assert!(!html.contains("javascript:"));
    }

    #[test]
    fn escapes_search_snippet_but_keeps_marks() {
        let out = escape_snippet("<script>x</script> <mark>hit</mark>");
        assert!(out.contains("&lt;script&gt;x&lt;/script&gt;"));
        assert!(out.contains("<mark>hit</mark>"));
    }

    #[test]
    fn strip_atx_h1_drops_first_heading() {
        let out = strip_leading_h1("# Title\n\nbody text\n");
        assert_eq!(out, "body text\n");
    }

    #[test]
    fn strip_atx_h1_tolerates_leading_blank_lines() {
        let out = strip_leading_h1("\n\n# Title\n\nbody\n");
        assert_eq!(out, "body\n");
    }

    #[test]
    fn strip_atx_h1_leaves_h2_alone() {
        let out = strip_leading_h1("## Subhead\n\nbody\n");
        assert_eq!(out, "## Subhead\n\nbody\n");
    }

    #[test]
    fn strip_atx_h1_leaves_body_without_title_alone() {
        let out = strip_leading_h1("just a paragraph\n");
        assert_eq!(out, "just a paragraph\n");
    }

    #[test]
    fn strip_setext_h1_drops_first_heading() {
        let out = strip_leading_h1("Title\n=====\n\nbody\n");
        assert_eq!(out, "body\n");
    }

    #[test]
    fn strip_does_not_eat_setext_h2() {
        // `----` underlines are H2, not H1. Leave them alone.
        let out = strip_leading_h1("Title\n----\n\nbody\n");
        assert_eq!(out, "Title\n----\n\nbody\n");
    }

    #[test]
    fn wikilink_resolves_against_current_project() {
        let html = render("see [[notes/foo]] here", "default", "scratch");
        assert!(
            html.contains(r#"<a href="/web/w/default/scratch/p/notes/foo.md">notes/foo</a>"#),
            "got: {html}"
        );
    }

    #[test]
    fn wikilink_label_and_md_suffix() {
        // explicit label wins; `.md` appended when the last segment is bare.
        let html = render("[[notes/foo|the foo]] and [[bar.md]]", "default", "scratch");
        assert!(html.contains(r#">the foo</a>"#), "label: {html}");
        assert!(
            html.contains(r#"href="/web/w/default/scratch/p/notes/foo.md""#),
            "suffix add: {html}"
        );
        assert!(
            html.contains(r#"href="/web/w/default/scratch/p/bar.md""#),
            "suffix keep: {html}"
        );
    }

    #[test]
    fn wikilink_cross_project_and_cross_workspace_scope() {
        let html = render(
            "[[otherproj:notes/x]] [[ws2/proj2:y]]",
            "default",
            "scratch",
        );
        assert!(
            html.contains(r#"href="/web/w/default/otherproj/p/notes/x.md""#),
            "project scope: {html}"
        );
        assert!(
            html.contains(r#"href="/web/w/ws2/proj2/p/y.md""#),
            "workspace/project scope: {html}"
        );
    }

    #[test]
    fn wikilink_not_linkified_in_code() {
        let fenced = render("```\n[[notes/foo]]\n```", "default", "scratch");
        assert!(!fenced.contains("<a href"), "fenced: {fenced}");
        assert!(fenced.contains("[[notes/foo]]"), "fenced literal: {fenced}");

        let inline = render("use `[[notes/foo]]` literally", "default", "scratch");
        assert!(!inline.contains("<a href"), "inline code: {inline}");
    }

    #[test]
    fn malformed_or_external_wikilink_kept_literal() {
        // unterminated
        let a = render("text [[notes/foo without close", "default", "scratch");
        assert!(a.contains("[[notes/foo without close"), "unterminated: {a}");
        // external scheme inside [[ ]] is not an internal page → literal
        let b = render("[[https://example.com]]", "default", "scratch");
        assert!(!b.contains("<a href"), "external: {b}");
        // path traversal rejected
        let c = render("[[../etc/passwd]]", "default", "scratch");
        assert!(!c.contains("<a href"), "traversal: {c}");
    }
}
