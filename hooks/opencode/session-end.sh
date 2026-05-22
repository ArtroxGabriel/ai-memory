#!/bin/sh
# opencode session-end hook.
SERVER="${AI_MEMORY_HOOK_URL:-http://127.0.0.1:7777}"
curl -s --max-time 2.0 \
    -X POST "$SERVER/hook?event=session-end&agent=open-code" \
    -H "Content-Type: application/json" \
    --data-binary @- >/dev/null 2>&1 || true
exit 0
