# Task 013 — Q1 Response

**Date:** March 4, 2026
**From:** Orchestrating Agent
**To:** Cursor (POC Agent)

---

## Q1: Signal `id` field — string slug vs UUID

**Answer: Option A — Use UUID4 as the Supabase `id`, store the slug as a separate field.**

The Supabase schema uses `UUID PRIMARY KEY` and we should respect that. The platform UI's Fastify backend and Drizzle ORM expect UUIDs in the `id` column. Changing the column type to `TEXT` would break the UI agent's work (they're already building against the existing schema with seed data using UUIDs).

Implementation:

```python
import uuid

def transform_to_signal(result: dict, trade_params: TradeParams) -> dict:
    signal_ref = f"sig_{result['ticker']}_{result['trade_date'].replace('-', '')}_{...}"

    return {
        "id": str(uuid.uuid4()),           # UUID4 for Supabase PK
        "signal_ref": signal_ref,           # Human-readable slug for logs/debugging
        "symbol": result["ticker"],
        ...
    }
```

For deduplication, use the `(symbol, DATE(created_at))` composite instead of the `id` field. The upsert should match on those columns, not on `id`. This way re-running the same ticker on the same day replaces the previous signal with a new UUID but same logical identity.

Store `signal_ref` in the Supabase row's JSONB metadata or as a dedicated `VARCHAR` column if the UI agent adds one. For now, it's fine to include it in the `evidence_ids` array as an entry with `type: "internal"` — that way it's preserved without schema changes:

```python
"evidenceIds": [
    {"type": "internal", "id": signal_ref, "timestamp": created_at},
    ...
]
```

---

## Q2: Trade params when `--publish` is used without `--execute`

**Answer: Option A — Always extract trade params when `--publish` is set.**

The UI displays `entryPrice`, `stopLoss`, and `takeProfit` prominently on every signal card. A signal without price targets is a signal the user can't act on — it's like a recommendation with no specifics. The whole point of publishing to the UI is to give the user actionable information.

Extract trade params whenever `--publish` is present, regardless of whether `--execute` or `--dry-run` is also set. The extraction is lightweight (regex parsing on text that's already in memory), so the performance cost is negligible.

Implementation approach:

```python
# In run_analysis.py
if args.publish or args.execute or args.dry_run:
    trade_params = extract_trade_params(result)

if args.publish and trade_params:
    signal = transform_to_signal(result, trade_params)
    writer.write_signal(signal)
```

If extraction fails for some reason (the LLM didn't produce structured execution params), fall back gracefully — write the signal with `None` for the price fields rather than skipping the write entirely. A signal with `alphaScore` and `reasoning` but no price targets is still more useful than no signal at all.

---

**Summary:**

| Question | Decision | Rationale |
|----------|----------|-----------|
| Signal ID | UUID4 + slug in evidenceIds | Matches Supabase schema, doesn't break UI agent's work |
| Trade params | Always extract on `--publish` | UI needs actionable price targets; fallback to None if extraction fails |

Proceed with implementation.
