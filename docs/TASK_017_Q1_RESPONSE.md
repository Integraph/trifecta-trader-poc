# Task 017 Q1 Response — Design Decisions

## Q1: Testing Requirements → TypeScript + Build Only (No Vitest)

The quality gate for Task 017 is:

1. `tsc --noEmit` passes (zero TypeScript errors)
2. `npm run build` completes without errors
3. All pages render and connect to the API correctly

No Vitest unit tests required. This is a frontend dashboard for a local admin tool — the real test is "does it work when you open it in a browser." The TypeScript compiler catches most structural issues. Writing React component tests for an internal tool at this stage would be low ROI.

If we add frontend tests later, it would be for the utility layer (`lib/utils.ts` formatters, `WebSocketManager` reconnect logic) — but that's a future concern.

---

## Q2: WebSocket URL Routing → Option B (Direct Connection)

**Choice: Option B — Connect directly to `ws://localhost:8420/...`**

Your reasoning is exactly right. This is a local-only admin tool with CORS wide open. There's no reason to proxy WebSocket traffic through Vite when the frontend can connect directly to the API server. It's simpler, more reliable, and avoids the proxy path mismatch entirely.

The `WebSocketManager` in `lib/websocket.ts` should connect to:
- `ws://localhost:8420/ws/events` for the event stream
- `ws://localhost:8420/logs/ws/logs` for the log stream

The Vite proxy rules for `/api` can stay as a convenience for REST calls during development, but the WebSocket connections go direct.

---

## Summary

| Question | Choice | Rationale |
|----------|--------|-----------|
| Q1: Testing | TypeScript + build only | Low ROI for internal admin tool; compiler catches structural issues |
| Q2: WebSocket routing | Option B — direct connection | Simplest, most reliable for local-only tool |
