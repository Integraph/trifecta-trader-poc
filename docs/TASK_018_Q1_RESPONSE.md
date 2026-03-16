# Task 018 — Question Responses

---

## Q1: Config Count — Migrate all 13

Yes, migrate **all 13**. The spec said "12" as a rough count — that was a miscount on my part. All configs in the `CONFIGS` dict should be migrated to `config/hybrid_llm.yaml`. None should be excluded.

Update exit criterion #2 accordingly: "Generated YAML contains all **13** existing configs with all 10 fields each."

---

## Q2: Providers List — Option A (hardcoded canonical list)

Agreed with your recommendation. **Option A — hardcoded canonical list.**

```python
KNOWN_PROVIDERS = ["anthropic", "ollama", "openai", "google", "xai", "openrouter"]
```

This is the correct choice because:
- The sanity check endpoint needs to know how to test each provider (different URLs, auth patterns) — this mapping is inherently hardcoded
- Validation on create/update should reject unknown providers, which requires a fixed set
- Option B would miss valid providers that happen to not be used in any existing config yet

If a new provider needs to be added in the future, it requires a code change to add the sanity check logic anyway, so adding it to the list at the same time is natural.

---

## Route Ordering Note

Good catch on the `/ab` vs `/{task_id}` ordering. Correct — register `POST /test-run/ab` and `GET /test-run/ab/{ab_id}` **before** the existing `GET /test-run/{task_id}` catch-all to prevent FastAPI from matching "ab" as a task_id.

---

No other questions needed. Proceed with implementation.
