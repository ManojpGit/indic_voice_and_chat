# Design: Post-call Lead Outcome Analysis

**Date:** 2026-06-05. **Status:** approved (brainstorm complete), ready for implementation plan.

## Problem

After a call we want to classify it into a standardized **Lead Call Outcome Type**, and produce a human-readable **summary** and **notes**. When the outcome is a callback request, also produce a resolved **callback date/time**. Today nothing does this: the campaign pipeline has scaffolding (`CallResult`, `CallDisposition`, a fake CRM, orchestrator retry/DND logic) but the "analyze a finished call → produce the result" step does not exist, and the dev console shows no outcome at all.

## The 10 outcome types

Conversational (require transcript analysis):
1. Interested
2. Callback requested
3. Not interested
4. Refused
5. Escalated
6. Angry / Hostile

Unreachable (come from telephony signaling, no transcript):
7. No answer
8. Voicemail
9. Busy
10. Call failed

## Decisions (from brainstorm)

- **Scope:** one shared, transport-agnostic analyzer that BOTH the dev console and the campaign pipeline call.
- **Taxonomy:** new canonical `LeadCallOutcome` enum (the 10 types) + a mapping to the legacy `CallDisposition` so the existing orchestrator/CRM/benchmark scaffolding keeps working unchanged.
- **Method:** a dedicated post-call LLM pass over the full transcript (+ collected slots), not derived from per-turn signals. Runs after hangup (user already gone, latency irrelevant).
- **Output language:** English (summary + notes), regardless of call language.
- **Callback time:** resolve relative phrases to an absolute, tz-aware datetime when possible; when vague/missing, leave `callback_datetime` null and capture the phrase in notes. Timezone comes from the **tenant config** (new field), not hardcoded.

## Architecture

### New module
- **`src/analysis/call_outcome.py`** (new) — the transport-agnostic core. Pure function/class; depends only on an LLM provider + plain inputs, so it is unit-testable with a fake LLM.

### Data shapes
- **`LeadCallOutcome`** (`str, Enum`) in `src/campaign/models.py`, alongside `CallDisposition`:
  `INTERESTED, CALLBACK_REQUESTED, NOT_INTERESTED, REFUSED, ESCALATED, ANGRY_HOSTILE, NO_ANSWER, VOICEMAIL, BUSY, CALL_FAILED`.
- **`CallAnalysis`** (Pydantic, in `src/campaign/models.py`):
  - `outcome: LeadCallOutcome`
  - `summary: str` (English, 2-3 sentences)
  - `notes: str` (English; objections, preferences, next-step hints; also holds the raw callback phrase when unresolved)
  - `callback_datetime: Optional[datetime]` (tz-aware)
  - `callback_phrase: Optional[str]` (raw, e.g. "kal shaam 5 baje")
  - `analysis_source: Literal["llm", "telephony", "fallback"]`

### Core flow — `analyze_call(...)`
Inputs: `transcript` (list of role/content turns), `slots` (collected), `telephony_status` (Optional), `final_action` (Optional, the agent's last action), `tenant_timezone` (str), `now` (call-end timestamp), `llm` (provider).

```
if telephony_status maps to an unreachable outcome (no_answer/busy/failed/voicemail):
    outcome = telephony map; summary = canned line; analysis_source = "telephony"; no LLM call
else:                       # a real conversation happened
    run ONE LLM call: transcript + slots + tenant tz + now
      → JSON {outcome ∈ 6 conversational, summary, notes, callback_datetime?, callback_phrase?}
    parse + validate; analysis_source = "llm"
on LLM failure / parse error / timeout:
    fallback: derive outcome from final_action
      (close_positive→INTERESTED, schedule_callback→CALLBACK_REQUESTED,
       transfer→ESCALATED, close_negative→NOT_INTERESTED, else NOT_INTERESTED)
    summary/notes = minimal, notes append "(auto-derived; LLM analysis failed)"
    analysis_source = "fallback"
```

The LLM call reuses the tenant's configured LLM provider (Gemini) with `response_format=json`, bounded by a timeout (reuse `TURN_TIMEOUT_S`-style ceiling, e.g. 15s).

## Mapping tables

### Telephony status → outcome
| Telephony status (normalized) | LeadCallOutcome |
|---|---|
| no_answer | NO_ANSWER |
| busy | BUSY |
| failed / canceled | CALL_FAILED |
| voicemail (AMD) | VOICEMAIL |

### LeadCallOutcome → legacy CallDisposition
Keeps the orchestrator (retry/DND/qualifying) + benchmarks working.
| LeadCallOutcome | CallDisposition | Orchestrator effect |
|---|---|---|
| INTERESTED | interested_transfer | qualifying |
| CALLBACK_REQUESTED | interested_callback | qualifying |
| NOT_INTERESTED | not_interested | complete |
| REFUSED | **dnd_requested** | mark DND |
| ESCALATED | interested_transfer | qualifying |
| ANGRY_HOSTILE | **dnd_requested** | mark DND |
| NO_ANSWER | busy_retry | schedule retry |
| BUSY | busy_retry | schedule retry |
| CALL_FAILED | busy_retry | schedule retry |
| VOICEMAIL | voicemail | schedule retry |

## Callback time resolution
- New `timezone: str = "Asia/Kolkata"` field on the tenant config model (`src/config_tenant.py`), set per-tenant in tenant YAML.
- The LLM prompt receives the tenant timezone and the call-end timestamp (`now`) and is instructed to return `callback_datetime` as an ISO-8601 tz-aware string, or null if the lead did not give a resolvable time.
- Resolvable ("kal shaam 5 baje") → absolute tz-aware datetime. Vague/missing ("call me later") → `callback_datetime=null`, phrase recorded in `callback_phrase`/notes.
- Parsing uses `zoneinfo` (stdlib).

## Integration

### Dev console (visible first)
- On call end (terminal turn or disconnect, in `BrowserVoiceBridge`), call `analyze_call` with the session transcript + slots (telephony_status = None → always the conversational path; pipeline error → CALL_FAILED).
- Push a new WS message: `{"type":"outcome", "outcome":..., "summary":..., "notes":..., "callback_datetime":..., "callback_phrase":...}`.
- Add a results panel to `static/dev_console.html`: outcome badge + summary + notes + callback line.

### Campaign pipeline
- Implement the missing CallResult-construction step: after a dispatched call returns, call `analyze_call`, then populate `CallResult`.
- Extend `CallResult` with: `outcome: LeadCallOutcome`, `summary`, `notes`, `callback_datetime`. `disposition` is derived from `outcome` via the mapping table (so existing consumers are untouched).
- Extend `Conversation` DB model (`src/models/conversation.py`) with `outcome` (String), `summary` (Text), `notes` (Text), `callback_at` (DateTime). Persistence stays optional — the local DB is not provisioned and the store may be None; the dev console path does not require DB writes.

## Error handling
- LLM failure / unparseable JSON / timeout → fallback path (above); never raises to the caller.
- Unknown/empty transcript with no telephony status → outcome from `final_action` fallback, or NOT_INTERESTED with a note if no signal at all.
- Bad timezone string → default to `Asia/Kolkata` with a logged warning.

## Testing
- **Mapping tables:** unit tests asserting both tables exhaustively (every enum member maps).
- **Analyzer (fake LLM):** canned JSON → assert `CallAnalysis` parse; assert callback resolution with a fixed `now` + tenant tz + "kal shaam 5 baje" → expected absolute datetime; assert vague phrase → `callback_datetime=null` + phrase captured.
- **Telephony short-circuit:** status=busy → BUSY, no LLM call made (assert the fake LLM was not invoked).
- **Fallback:** LLM raises → outcome derived from `final_action`, `analysis_source="fallback"`.
- **Dev console:** bridge emits an `outcome` WS message on call end (existing bridge test harness).

## YAGNI / out of scope
- No new/separate LLM provider — reuse the tenant's configured LLM.
- No AMD/voicemail detection implementation now — only the status→outcome mapping hook (the dev console cannot produce voicemail anyway).
- No DB migration tooling beyond the column additions; persistence remains optional until the campaign path is exercised live.
- No CRM adapter beyond the existing fake.

## Relevant existing code
- `src/campaign/models.py` — `CallDisposition`, `CallResult`.
- `src/campaign/orchestrator.py` — `_QUALIFYING/_RETRY/_DND_DISPOSITIONS`, consumes `result.disposition`.
- `src/integration/crm_client.py` — `update_lead(call_result)` (fake).
- `src/dialogue/response_parser.py` — per-turn `action`, `sentiment`, `internal_notes` (fallback source).
- `src/api/browser_bridge.py` — dev console call lifecycle (`run()` end).
- `src/config_tenant.py` — tenant config model (timezone field goes here).
- `static/dev_console.html` — results panel.
