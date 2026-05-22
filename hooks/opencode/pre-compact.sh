#!/bin/sh
# opencode pre-compact hook.
SERVER="${AI_MEMORY_HOOK_URL:-http://127.0.0.1:7777}"
curl -s --max-time 0.5 \
    -X POST "$SERVER/hook?event=pre-compact&agent=open-code" \
    -H "Content-Type: application/json" \
    --data-binary @- >/dev/null 2>&1 || true
exit 0
