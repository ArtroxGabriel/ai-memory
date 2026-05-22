#!/bin/sh
# opencode post-tool-use hook.
SERVER="${AI_MEMORY_HOOK_URL:-http://127.0.0.1:7777}"
curl -s --max-time 0.5 \
    -X POST "$SERVER/hook?event=post-tool-use&agent=open-code" \
    -H "Content-Type: application/json" \
    --data-binary @- >/dev/null 2>&1 || true
exit 0
