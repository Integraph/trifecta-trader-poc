# Task 017: Admin Dashboard Frontend — Pre-Implementation Questions

---

## Q1: Testing Requirements

The 48 exit criteria make no mention of React/TypeScript tests.

Is the quality gate limited to:
- `tsc --noEmit` passes (zero TypeScript errors)
- `npm run build` completes without errors
- Manual visual verification in the browser

...or do you want any Vitest unit tests written as well (e.g., for `lib/utils.ts` formatters, the `WebSocketManager` class, or the custom `usePolling` hook)?

---

## Q2: WebSocket URL Routing in Dev Mode

The Vite dev server proxy in the spec defines:

```typescript
'/ws': { target: 'ws://localhost:8420', ws: true }
```

This correctly proxies `/ws/events` (the events WebSocket). However, the log stream
lives at `/logs/ws/logs` (the logs router is mounted with prefix `/logs` in `app.py`),
which does **not** match the `/ws` proxy rule and would fail in dev mode.

Three options:

**Option A — Add a second proxy rule in `vite.config.ts`:**
```typescript
'/logs/ws': { target: 'ws://localhost:8420', ws: true }
```
Minimal change, consistent with the spec's proxy approach.

**Option B — Bypass the proxy; connect directly to `ws://localhost:8420/...`:**
The frontend connects to `ws://localhost:8420/ws/events` and `ws://localhost:8420/logs/ws/logs`
directly (no proxy path). Works seamlessly since CORS is open and this is a local-only
tool. The `/api` and `/ws` proxy rules become optional convenience aliases.

**Option C — Move the logs WebSocket to a top-level `/ws/logs-stream` route:**
Change `src/admin/logs.py` so the WebSocket is registered outside the `/logs` router
prefix, making both WebSocket paths (`/ws/events`, `/ws/logs-stream`) catchable by
the single `/ws` proxy rule. Requires a minor backend change.

**My recommendation:** Option B — direct connection avoids proxy complexity entirely
and is the most reliable for a local admin tool. Option A is the smallest change if
you prefer to keep all traffic going through the Vite proxy.
