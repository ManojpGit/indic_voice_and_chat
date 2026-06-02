# Customer-Led Agent + Campaign-Upfront Script Loading — Design

**Date:** 2026-06-02
**Status:** Approved (pending spec review)
**Author:** Manoj + Claude

## Problem

Two things are hardwired today:

1. **The script is a hardcoded constant.** `DEFAULT_DEMO_SCRIPT` (the "Priya / Vox Demo" persona) is baked into `src/bootstrap.py` and used by every bridge. There is a `config/campaigns/*.yaml` convention and a `VoiceBotScript.from_campaign_yaml()` parser, but nothing loads a campaign file — the parser is dead code.

2. **The agent behaves like a script-reader.** The current system prompt lists talking points and qualifying questions; it does not tell the agent to *listen to the customer and answer what they actually asked first*. The desired behavior is customer-led: address the customer's real question/concern first, then gently advance the campaign objective, and only redirect when the input is totally unrelated.

We also have a richer real script (Anaaya / Bharat Matka) whose fields (`knowledge`, `objective`, `dos`, `donts`, `personality`, `gender`, `conversation_style`, `max_turns`) the current `VoiceBotScript` does not model.

## Goal

Load one campaign script at startup (campaign-upfront), selectable by config, and make the agent customer-led: answer the customer's actual question first (using the script's knowledge), then advance the objective, redirecting only on off-topic input. Ship the Bharat Matka / Anaaya campaign as the active script and let the dev console exercise it.

## Decisions (settled during brainstorming)

- **Resolution model: campaign-upfront, one script per process**, selectable via config. (Not per-call / CRM-resolved — deferred. The external CRM will own campaigns later; this is the single-campaign stepping stone.)
- **Reuse the existing `config/campaigns/*.yaml` + `from_campaign_yaml` pattern** rather than inventing a new config shape.
- **`max_turns` is a soft prompt guide**, not a hard stop.
- **Behavioral contract approved** as in "Behavioral contract" below.

## Non-Goals (YAGNI)

- Per-call / CRM-driven script resolution, multiple concurrent campaigns, hot-reload of the script file.
- Hard turn-count enforcement (soft guide only).
- Slot schema for this campaign (the Anaaya script defines no slots; the agent continues with an empty `SlotSchema()`). Can be added later.
- Pipeline overrides from the campaign YAML (voice_id/temperature) — out of scope; tenant config still drives providers.

## Background: existing pieces this builds on

- `src/dialogue/prompts.py`: `VoiceBotScript` dataclass + `from_campaign_yaml()` + `build_voicebot_system_prompt()` (+ `VOICEBOT_RESPONSE_SCHEMA`).
- `src/bootstrap.py`: `DEFAULT_DEMO_SCRIPT`, and `make_bridge_factory(...)` / `make_exotel_bridge_factory(...)` which take a `script: VoiceBotScript = DEFAULT_DEMO_SCRIPT`.
- `src/api/dev_console.py`: `make_browser_bridge_factory(..., script=DEFAULT_DEMO_SCRIPT)`.
- `src/main.py`: lifespan builds the provider registry and registers the bridge factories.
- `config/campaigns/sample_campaign.yaml`: existing campaign-YAML shape with a `campaign.script:` block.

## Architecture

```
config/campaigns/bharat_matka.yaml
        │  (campaign.agent + campaign.script blocks)
        ▼
load_campaign_script("bharat_matka")  ──▶ VoiceBotScript (extended fields)
        │  (selected by VOX_CAMPAIGN env, default "bharat_matka")
        ▼
main.py lifespan: load once  ──▶ pass `script=` into:
        ├─ make_bridge_factory (Twilio)
        ├─ make_exotel_bridge_factory
        └─ make_browser_bridge_factory (dev console)
        ▼
VoiceBotAgent → build_voicebot_system_prompt(script, ...)  ← customer-led rewrite
```

### Components

| File | Change |
|------|--------|
| `config/campaigns/bharat_matka.yaml` | **Create.** Anaaya / Bharat Matka campaign in the existing campaign-YAML shape (a `campaign:` wrapper with `agent:` and `script:` blocks). |
| `src/dialogue/prompts.py` | **Modify.** Extend `VoiceBotScript` with new fields; extend `from_campaign_yaml()` to parse them (tolerant of `greeting`/`opening`, `name`/`agent_name`, etc.); rewrite `build_voicebot_system_prompt()` for the customer-led contract. |
| `src/dialogue/campaign_loader.py` | **Create.** `load_campaign_script(slug, campaigns_dir=...) -> VoiceBotScript` and `active_campaign_slug() -> str` (reads `VOX_CAMPAIGN`, default `bharat_matka`). |
| `src/main.py` | **Modify.** In lifespan, load the active campaign script once and pass it into all three bridge factories. |
| `src/bootstrap.py` / `src/api/dev_console.py` | No signature change needed — they already accept `script=`. Lifespan passes the loaded script. |

### Extended `VoiceBotScript`

New optional fields (all default empty so existing callers/tests are unaffected):

```python
personality: str = ""
gender: str = ""
objective: str = ""
knowledge: dict[str, str] = {}        # concern/topic -> reference answer
dos: list[str] = []
donts: list[str] = []
conversation_style: str = ""
max_turns: int = 0                    # 0 = no soft budget
```

`closing` stays `dict[str, str]`, but `from_campaign_yaml` accepts a **string** closing too (wraps it as `{"default": <str>}`) since the Anaaya script uses a single closing line.

`from_campaign_yaml` field tolerance (accept either key):
- `agent_name` or `name`; `company_name` or `company`; `agent_role` or `role`; `language_default` or `language`; `opening` or `greeting`.
- New keys read directly: `personality`, `gender`, `objective`, `knowledge`, `dos`, `donts`, `conversation_style`, `max_turns`.

### `load_campaign_script`

```
active_campaign_slug(): return os.environ.get("VOX_CAMPAIGN", "bharat_matka")

load_campaign_script(slug, campaigns_dir=Path("config/campaigns")):
    path = campaigns_dir / f"{slug}.yaml"
    if not path.exists(): log warning; return DEFAULT_DEMO_SCRIPT   # safe fallback
    data = yaml.safe_load(path)
    camp = data.get("campaign", data)            # tolerate with/without wrapper
    merged = {**(camp.get("agent") or {}), **(camp.get("script") or {})}
    return VoiceBotScript.from_campaign_yaml(merged)
```

The `agent:` block (name/company/role/personality/language/gender) is merged with the `script:` block before parsing, so the campaign file can keep the two-block layout the user authored.

### `config/campaigns/bharat_matka.yaml`

A `campaign:` wrapper containing `agent:` and `script:` blocks, holding the full Anaaya content the user provided: greeting, objective, talking_points, knowledge (safety / scam_concerns / withdrawal / deposit / transaction_speed / support / referral), closing, dos, donts, max_turns: 12, conversation_style. Template tokens `{agent_name}` / `{lead_name}` continue to be substituted by the existing `_template_vars` path in `VoiceBotAgent.play_opening`.

## Behavioral contract (the customer-led rewrite)

`build_voicebot_system_prompt` is rewritten so the prompt instructs, in priority order:

1. **Listen first, answer first.** Each turn, first understand what the customer actually said and address *that* directly and helpfully, drawing on the `knowledge` entries in the agent's own warm words — never a canned recital.
2. **Then gently advance** the `objective` (build trust → mention "Official" status → the **10% bonus hook** → offer to send the WhatsApp link). Talking points are *available material, not a checklist*.
3. **Redirect only when the input is totally unrelated** (weather, wrong number, personal chit-chat): briefly, warmly acknowledge, then steer back to the conversation.
4. **Tone & style** from `personality` / `conversation_style` / `dos` / `donts`: high-energy, warm, persuasive, colloquial Hinglish; human fillers ("Actually", "Dekhiye", "Bas"); simple words (avoid heavy Hindi like "Jankari"); end turns with a soft prompt/question; never argue when the customer says "scam" — reassure calmly; never forget the 10% bonus.
5. **Soft turn budget** (`max_turns`): if the customer clearly is not engaging after a few attempts, close gracefully rather than pushing forever. Stated as guidance, not enforced in code.

The prompt still embeds `VOICEBOT_RESPONSE_SCHEMA` and the existing rules (concise 1–2 sentence `response_text`, honest about being AI, don't invent facts, `action` semantics). The per-turn JSON contract and the state machine are **unchanged** — `action` still drives transitions. This is a content/prompt change, not an architectural one.

## Data flow (unchanged loop, new content)

Per turn: STT → `build`-prompt-backed LLM call → structured JSON (`response_text`, `action`, `updated_slots`, `sentiment`, `conversation_phase`) → `handle_turn` applies slots + maps `action` to a state event → TTS. The only differences are the richer script content and the customer-led prompt guidance.

## Error handling

- Missing/unreadable campaign file → log a warning and fall back to `DEFAULT_DEMO_SCRIPT` so the app still boots and the console still works.
- Malformed YAML → `from_campaign_yaml` is lenient (uses `.get` with defaults); unknown keys are ignored; missing optional fields default to empty.
- `closing` as either string or dict is normalized.

## Testing

- **`from_campaign_yaml`**: parses the new fields; maps `greeting`→`opening`, `name`→`agent_name`, etc.; normalizes a string `closing`. (`tests/unit/test_prompts.py` or a new `test_campaign_script.py`.)
- **`load_campaign_script`**: returns a populated `VoiceBotScript` for `bharat_matka` (agent_name == "Anaaya", company_name == "Bharat Matka", knowledge non-empty, max_turns == 12); falls back to `DEFAULT_DEMO_SCRIPT` for a missing slug.
- **`build_voicebot_system_prompt`**: for the Anaaya script, the prompt contains the answer-first directive, the knowledge entries, the dos/donts, and the 10% bonus hook; and still contains the JSON schema.
- **Manual (console):** restart with the campaign active → talk to Anaaya → she answers your actual question first, weaves in the bonus, and only redirects on off-topic input.

## Success criteria

1. With `VOX_CAMPAIGN=bharat_matka` (the default), the console and telephony bridges run the Anaaya script; the opening greeting is Anaaya's.
2. The system prompt for that script demonstrably encodes the customer-led / answer-first contract and the campaign knowledge (verified by a unit test asserting key phrases).
3. Existing callers and tests are unaffected (new `VoiceBotScript` fields are optional; `DEFAULT_DEMO_SCRIPT` still valid).
4. A missing campaign file degrades gracefully to the demo script with a logged warning.
