# Terser replies → faster first word — design

**Date:** 2026-06-08. **Status:** approved. **Scope:** reduce perceived time-to-first-spoken-word
by prompting the agent to **lead each turn-reply with a short first sentence**. Prompt-only; no code
behavior change. Latency is inference-bound (see `docs/latency-llm-stt-experiments.md`); this targets
the one remaining safe-ish lever after the Singapore deploy and the Sarvam-streaming NO-GO.

## Goal
Cut `tts_first_ms` (and the perceived "pause after I stop talking") by feeding the existing overlapped
TTS pipeline a smaller first bite, **without** changing the LLM model, the TTS voice, or overall reply
length. Goal chosen over "shorter overall" because overlapped streaming hides the reply tail — only the
first sentence drives first-word latency.

## Mechanism (why prompt-only is enough)
The engine already streams the LLM JSON, extracts spoken text incrementally (`_SpokenTextExtractor`),
and `SentenceDetector` flushes each complete sentence to Sarvam the moment it arrives
(`src/pipeline/engine.py` ~314-328). So a short **first** sentence → Sarvam starts sooner and synthesizes
less → lower `tts_first_ms`. No change to the parser, detector, or engine.

## The change
In `build_voicebot_system_prompt` (`src/dialogue/prompts.py`, the "Rules:" block ~line 273), add a rule:

- Begin every reply with a **short first sentence** (~2-5 words) — a brief, natural acknowledgment —
  ending in a sentence boundary (`।`/`.`), *then* continue with any detail or question.
- **Vary** the opener across turns (rotate, e.g. जी / अच्छा / बिल्कुल / समझ गई / हाँ); never reuse the
  same opener mechanically — it must sound warm and human, not scripted.

The existing "Keep `response_text` concise (1-2 sentences)" rule stays.

## Scope & guardrails
- **Turn replies only** — not the pre-rolled opening greeting (already spoken, built separately).
- Don't bolt a filler onto a reply that is already short (e.g. "जी बोलिए।").
- It is a **sales/persuasion bot**: warmth and persuasiveness must not regress. Variety is the
  anti-repetition safeguard.

## Testing
- **Unit:** a presence test that `build_voicebot_system_prompt(...)` includes the new lead-short-opener
  rule text (prompts are LLM-driven; behavior can't be unit-tested, but the rule's presence can).
- **Live A/B (the real gate):** capture ~8-10 dev-console turns on current main (BEFORE) and after the
  change (AFTER); compare median/mean `tts_first_ms` and `total_ms` from the `"browser turn (stream)"`
  log lines; **listen** for warmth and opener repetition. The win is a hypothesis (`tts_first_ms` is
  partly fixed: `llm_ttft` ~1.4s + Sarvam base first-chunk), so the A/B decides.

## Decision rule
Keep the change only if it **measurably** lowers `tts_first_ms` **without** a quality/warmth regression.
Record the dated before/after result in `docs/latency-llm-stt-experiments.md`. If no measurable gain or
quality drops → revert (pure prompt text, trivially reversible).
