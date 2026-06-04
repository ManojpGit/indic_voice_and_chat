# Session Handoff — Barge-in (IN PROGRESS) + session state

**Date:** 2026-06-04. **Read this first on resume.** It captures uncommitted working-tree state and exactly where the in-flight barge-in work stands.

---

## TL;DR — what to do on resume

Barge-in is **built, reviewed, and mostly working**; we're in the **live tune/validate** loop. The user last reported: barge-in fires and addresses the interruption, but it sometimes **listened too long / got stuck**. That stuck/slow bug was root-caused and **fixed** (commit `6bc0ce4` — stale Deepgram `_endpointed` flag). **Awaiting the user's retest of that fix.**

**Immediate next steps:**
1. Ask the user to retest barge-in at **http://localhost:8765/dev/voice** (NOT ngrok — see Operational below) and confirm: no more stuck, delay-before-thinking acceptable.
2. If good → **strip the DIAG logging** (see "Uncommitted state" below — exact lines), **commit the tuned `dev_console.html`**, run `.venv/bin/python -m pytest tests/unit -q` (expect 628+ pass), then **finish the branch** (decision log update + push).
3. If still self-triggering on echo → raise `BARGE_RMS` toward 0.007. If still missing interruptions → lower toward 0.005. If still slow-but-not-stuck → it's Deepgram falling back to the 1000ms `UtteranceEnd` post-gap (its minimum; can't lower) — discuss keeping the stream warm during agent turns as a follow-up.

---

## Uncommitted working-tree state (LOST on restart if not handled)

`git status` shows two modified files (plus untracked `.claude/` which is intentionally never committed):

### `static/dev_console.html` — MIX of keepers + DIAG
**KEEP (real tuning, validated):**
- `const BARGE_RMS = 0.006;` (was 0.02 — measured: user speech ~0.008–0.012 RMS post-AEC, echo floor <0.005).
- The decay accumulator: `bargeMs = rms > BARGE_RMS ? bargeMs + frameMs : Math.max(0, bargeMs - frameMs);` (was a hard reset to 0).

**STRIP (DIAG throwaway):**
- `window._bargeN = 0;   // DIAG throttle counter` line.
- The two `console.log` DIAG lines in the worklet handler: `[barge] rms=...` (the `if (rms > 0.005 && (++window._bargeN % 5 === 0)) console.log(...)`) and `[barge] FIRED ...`.

### `src/api/browser_bridge.py` — ALL DIAG (strip entirely)
Three added `log.info("DIAG ...")` lines:
- In `run()`'s `barge_in` branch: `log.info("DIAG barge_in frame received", ...)` and the `elif ctrl: log.info("DIAG control frame", ...)` branch.
- In `_handle_barge_in()`: `log.info("DIAG barge-in ignored (agent not busy)")` before the early `return`.
Revert these so `_handle_barge_in` and the `run()` text branch match their committed form (the committed `run()` branch was just `if ctrl.get("type")=="barge_in": self._handle_barge_in()` then `continue`).

> Note: `src/providers/stt/deepgram.py` has a kept observability line `log.warning("deepgram stream ended unexpectedly", ...)` — that is COMMITTED (931c277), not DIAG, leave it.

---

## Barge-in feature status (the in-flight work)

- **Spec:** `docs/superpowers/specs/2026-06-04-barge-in-design.md` (committed `dceb70a`).
- **Plan:** `docs/superpowers/plans/2026-06-04-barge-in.md` (committed `e609a56`).
- **Approach:** browser-side detection (VAD on echo-cancelled mic) + instant `stopPlayback()` + `{"type":"barge_in"}` → server `_handle_barge_in()` sets the in-flight turn's `cancel_event` + clears `_agent_busy` → engine stops LLM/TTS → `handle_turn_text` records the user turn, drops the abandoned reply, → LISTENING → interruption becomes the next turn. The cancel/turn-handling core is transport-agnostic (telephony reuses it later via a server-side detector calling the same `_handle_barge_in()`).
- **Implemented & reviewed (subagent-driven, all on `main`):**
  - `cfbae82` engine cancel-stops-audio test.
  - `db657e1` voicebot: `handle_turn_text(cancel_event=)`; cancelled turn keeps user turn, drops reply, → LISTENING (`parse_error="barge-in"`).
  - `3360de2` bridge: `_handle_barge_in()`, per-turn `cancel_event` in `_dispatch_text_turn`, `barge_in` routing.
  - `c7e4fe4` browser: `activeSources`/`stopPlayback()`, VAD detect, "Allow interruptions" toggle.
  - Final holistic review: **Ready to merge.** 627 tests passed at that point.
- **Bug found in live test + FIXED:** `6bc0ce4` — after a `speech_final`, Deepgram session's `_endpointed` stayed True and suppressed the next utterance's `UtteranceEnd` backup → the post-barge-in interruption could get **stuck in listening** (or slow via the 1000ms UtteranceEnd). Fix clears `_endpointed` when new speech arrives. Has a regression test. 628 tests pass.
- **Remaining:** confirm the fix live → strip DIAG → commit tuned `dev_console.html` → push.

---

## Git state

- **Local `main` HEAD:** `6bc0ce4`.
- **`origin/main`:** behind — these commits are **NOT pushed yet**: `c630d46` (turn-timeout fix), `931c277` (streaming-STT crash/drop logging), `dceb70a`+`e609a56` (barge-in spec+plan), `cfbae82` `db657e1` `3360de2` `c7e4fe4` (barge-in impl), `6bc0ce4` (deepgram endpoint fix), plus the docs commits from earlier (`3476f51`, `49a6a25` were pushed; verify). **Push after barge-in is finalized** (user has been asked to confirm before pushing).
- Untracked `.claude/` — never commit.

---

## Operational (how to run / test)

- **Run the dev server:**
  `VOX_DEV_CONSOLE=1 .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env`
  (The repo's venv is `.venv`; tests/run use it. `anthropic`, `deepgram-sdk` are installed there and pinned in `pyproject.toml`.)
- **Test the console in the browser:** **http://localhost:8765/dev/voice** — use localhost, NOT ngrok. `getUserMedia` treats localhost as a secure context, so the mic works; no tunnel hop = lower latency. ngrok is only needed for telephony webhooks.
- **ngrok is currently DOWN** (ERR_NGROK_3200). Only restart it if doing a real phone call:
  `ngrok http --url=conceitedly-sinewy-margurite.ngrok-free.dev 8765`
- **Active config (`config/tenants/dev.yaml`):** LLM = `gemini-2.5-flash`; STT batch = Groq; **STT streaming = Deepgram `nova-2 hi`, `endpointing=300`, `utterance_end_ms=1000`** (the dev console uses streaming). Keys in gitignored `.env`: `TENANT_DEV_{GEMINI,ANTHROPIC,DEEPGRAM,GROQ,SARVAM}_KEY`.
- **Per-turn metrics** in the server log: `"browser turn (stream)"` lines carry `llm_ttft_ms/llm_total_ms/tts_first_ms/total_ms/action/user_text/agent_text`. Barge-in logs `barge-in: cancelling current turn`.
- **Note:** the dev server is currently running (PID was 30523); after a machine/session restart it will be down — relaunch with the command above.

---

## Bigger-picture context (already committed/documented)

- **Decision log** (Claude-vs-Gemini, cloud latency, Deepgram streaming, bottleneck = LLM): `docs/latency-llm-stt-experiments.md`; index at `docs/README.md`.
- **Deepgram streaming STT** feature: shipped + pushed earlier (interface `src/interfaces/stt.py`, adapter `src/providers/stt/deepgram.py`, registry `STREAMING_STT_PROVIDERS`, engine `run_turn_text`, bridge streaming path). Kept ON as the dev-console default; quality/reliability win, ~0.7s off the pre-response gap.
- **Turn-timeout safety net** (`c630d46`): every turn bounded by `TURN_TIMEOUT_S=20.0` (`src/agents/voicebot.py`) so a hung provider can't wedge the agent.
- **Known limitations / standing decisions:** kept Gemini (Claude was a latency wash + had a 36s stall; Claude adapter available behind `provider: anthropic|claude`). Latency floor is LLM-inherent (~2.0–2.5s to first word). Don't leave the dev server running for days — a stale Gemini client caused a latency creep that a restart fixed.
