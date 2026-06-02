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
- **Slots are declared per-campaign in config** (parsed by the existing `SlotSchema.from_campaign_yaml`). The agent fills whatever slots a campaign declares; the system assumes no specific slot names.

## Core principle: the logic is campaign-agnostic

The Bharat Matka / Anaaya script is **sample data, not a template baked into code.** No campaign-specific content (e.g. "10% bonus", "WhatsApp link", any specific slot name) ever appears in `build_voicebot_system_prompt`, the loader, or the dialogue loop. The system only ever consumes *whatever a campaign provides* — its `objective`, `knowledge`, `talking_points`, `dos`/`donts`, `closing`, and declared `slots`. The customer-led *policy* (answer-first → advance the objective → redirect only when off-topic → tone from the script's own `personality`/`dos`/`donts`) is generic dialogue behavior that holds for every campaign. Tests assert this *mechanism* (whatever the fixture campaign declares shows up), using Bharat Matka only as a fixture — never by hardcoding its strings into the logic.

## Non-Goals (YAGNI)

- Per-call / CRM-driven script resolution, multiple concurrent campaigns, hot-reload of the script file.
- Hard turn-count enforcement (soft guide only).
- Auto-discovery/inference of slots from script text (slots are explicitly declared in each campaign's config).
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
        │  (campaign.agent + campaign.script + campaign.slots blocks)
        ▼
load_campaign("bharat_matka")  ──▶ LoadedCampaign(script: VoiceBotScript, slots: SlotSchema)
        │  (selected by VOX_CAMPAIGN env, default "bharat_matka")
        ▼
main.py lifespan: load once  ──▶ pass `script=` AND `slots=` into:
        ├─ make_bridge_factory (Twilio)
        ├─ make_exotel_bridge_factory
        └─ make_browser_bridge_factory (dev console)
        ▼
VoiceBotAgent(script, slot_schema=slots) → build_voicebot_system_prompt(script, schema, ...)
        ← customer-led rewrite, generic over whatever the script/slots declare
```

### Components

| File | Change |
|------|--------|
| `config/campaigns/bharat_matka.yaml` | **Create.** Anaaya / Bharat Matka campaign in the existing campaign-YAML shape (a `campaign:` wrapper with `agent:`, `script:`, and `slots:` blocks). The slot set is sample data — a sensible discovered set for this campaign. |
| `src/dialogue/prompts.py` | **Modify.** Extend `VoiceBotScript` with new fields; extend `from_campaign_yaml()` to parse them (tolerant of `greeting`/`opening`, `name`/`agent_name`, etc.); rewrite `build_voicebot_system_prompt()` for the customer-led contract — generic over whatever the script declares. |
| `src/dialogue/campaign_loader.py` | **Create.** `LoadedCampaign` dataclass (`script`, `slots`), `load_campaign(slug, campaigns_dir=...) -> LoadedCampaign`, and `active_campaign_slug() -> str` (reads `VOX_CAMPAIGN`, default `bharat_matka`). |
| `src/bootstrap.py` | **Modify.** `make_bridge_factory` / `make_exotel_bridge_factory` gain a `slots: SlotSchema = SlotSchema()` param and pass it into `VoiceBotAgent(slot_schema=...)` instead of the hardcoded `SlotSchema()`. |
| `src/api/dev_console.py` | **Modify.** `make_browser_bridge_factory` gains the same `slots: SlotSchema = SlotSchema()` param and passes it into the agent. |
| `src/main.py` | **Modify.** In lifespan, `load_campaign(active_campaign_slug())` once and pass `script=` and `slots=` into all three bridge factories. |

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

### Per-campaign slots

Slots are declared in the campaign YAML's `slots:` block and parsed by the **existing** `SlotSchema.from_campaign_yaml(slots_dict)` (already in `src/dialogue/slots.py`; maps each entry to a `SlotSpec` with `type`/`required`/`values`). No new slot-parsing code is needed. The factories — which currently hardcode `SlotSchema()` — instead receive the campaign's `SlotSchema` and pass it into `VoiceBotAgent(slot_schema=...)`. The prompt's "Slots to fill" section is already generic (it iterates `schema.specs`), so declared slots automatically appear in the prompt and the agent fills them via the existing `updated_slots` path. The system never references specific slot names.

### `load_campaign`

```
active_campaign_slug(): return os.environ.get("VOX_CAMPAIGN", "bharat_matka")

@dataclass
class LoadedCampaign:
    script: VoiceBotScript
    slots: SlotSchema

load_campaign(slug, campaigns_dir=Path("config/campaigns")) -> LoadedCampaign:
    path = campaigns_dir / f"{slug}.yaml"
    if not path.exists():
        log warning
        return LoadedCampaign(DEFAULT_DEMO_SCRIPT, SlotSchema())   # safe fallback
    data = yaml.safe_load(path)
    camp = data.get("campaign", data)            # tolerate with/without wrapper
    merged = {**(camp.get("agent") or {}), **(camp.get("script") or {})}
    script = VoiceBotScript.from_campaign_yaml(merged)
    slots = SlotSchema.from_campaign_yaml(camp.get("slots") or {})
    return LoadedCampaign(script, slots)
```

The `agent:` block (name/company/role/personality/language/gender) is merged with the `script:` block before parsing, so the campaign file can keep the two-block layout the user authored. The `slots:` block is parsed independently.

### `config/campaigns/bharat_matka.yaml`

A `campaign:` wrapper containing `agent:`, `script:`, and `slots:` blocks:
- `agent:` / `script:` — the full Anaaya content the user provided: greeting, objective, talking_points, knowledge (safety / scam_concerns / withdrawal / deposit / transaction_speed / support / referral), closing, dos, donts, max_turns: 12, conversation_style.
- `slots:` — a discovered sample set for this campaign (e.g. `interest_level` (enum), `wants_whatsapp_link` (boolean), `deposit_intent` (boolean), `main_objection` (enum: scam/safety/withdrawal/none), `callback_time` (datetime)). This is *example data* the user can edit; the system makes no assumptions about these names.

Template tokens `{agent_name}` / `{lead_name}` continue to be substituted by the existing `_template_vars` path in `VoiceBotAgent.play_opening`.

## Behavioral contract (the customer-led rewrite)

`build_voicebot_system_prompt` is rewritten so the prompt instructs the agent, in priority order. **This is generic policy** — it references the script's *fields*, never any campaign's specific content. The Bharat Matka examples in parentheses below illustrate what *that campaign's data* fills in, but they are never written into the builder:

1. **Listen first, answer first.** Each turn, first understand what the customer actually said and address *that* directly and helpfully, drawing on the script's `knowledge` entries in the agent's own warm words — never a canned recital.
2. **Then gently advance the script's `objective`.** Use the script's `talking_points` as *available material, not a checklist.* (For Bharat Matka the objective happens to be: build trust → "Official" status → the bonus → offer the WhatsApp link — but that text comes entirely from the YAML.)
3. **Redirect only when the input is totally unrelated** (weather, wrong number, personal chit-chat): briefly, warmly acknowledge, then steer back to the conversation.
4. **Tone & style come from the script's own `personality` / `conversation_style` / `dos` / `donts`** — the builder embeds whatever those fields say and tells the agent to follow them. (Bharat Matka's happen to be: high-energy, warm, persuasive Hinglish; human fillers; simple words; end turns with a soft prompt; never argue on "scam"; always surface the bonus.)
5. **Soft turn budget** from the script's `max_turns` (if > 0): if the customer clearly is not engaging after a few attempts, close gracefully rather than pushing forever. Stated as guidance, not enforced in code.

The prompt still embeds `VOICEBOT_RESPONSE_SCHEMA` and the existing rules (concise 1–2 sentence `response_text`, honest about being AI, don't invent facts, `action` semantics) and the generic "Slots to fill" section driven by `schema.specs`. The per-turn JSON contract and the state machine are **unchanged** — `action` still drives transitions. This is a content/prompt change, not an architectural one.

## Data flow (unchanged loop, new content)

Per turn: STT → `build`-prompt-backed LLM call → structured JSON (`response_text`, `action`, `updated_slots`, `sentiment`, `conversation_phase`) → `handle_turn` applies slots + maps `action` to a state event → TTS. The only differences are the richer script content and the customer-led prompt guidance.

## Error handling

- Missing/unreadable campaign file → log a warning and fall back to `DEFAULT_DEMO_SCRIPT` so the app still boots and the console still works.
- Malformed YAML → `from_campaign_yaml` is lenient (uses `.get` with defaults); unknown keys are ignored; missing optional fields default to empty.
- `closing` as either string or dict is normalized.

## Testing

Tests assert the **generic mechanism** using a fixture campaign — they do not bake campaign strings into the production code, only into fixtures/assertions.

- **`from_campaign_yaml`**: parses the new fields; maps `greeting`→`opening`, `name`→`agent_name`, etc.; normalizes a string `closing` into a dict.
- **`SlotSchema.from_campaign_yaml`** (existing): a `slots:` dict yields the declared specs (covered by a small test feeding a sample slots block and asserting the spec names/types/required come back).
- **`load_campaign`**: for `bharat_matka` returns a `LoadedCampaign` whose `script` is populated (agent_name == "Anaaya", company_name == "Bharat Matka", `knowledge` non-empty, `max_turns` == 12) **and** whose `slots` contains the slots declared in that file (assert the declared names are present — read them from the file, don't hardcode an expectation that the system depends on). Missing slug → `LoadedCampaign(DEFAULT_DEMO_SCRIPT, SlotSchema())`.
- **`build_voicebot_system_prompt` is generic**: build the prompt from an *arbitrary* fixture script (NOT Bharat Matka) whose `objective`/`knowledge`/`dos`/`donts` contain unique sentinel strings, and assert those sentinels appear — proving the builder embeds whatever the script provides. Separately assert the fixed *policy* text is present (the answer-first instruction, the "redirect only when unrelated" instruction) and the JSON schema. No Bharat-Matka literal is required for the builder to pass.
- **Manual (console):** restart with the campaign active → talk to Anaaya → she answers your actual question first, advances the campaign's objective, only redirects on off-topic input, and the console's slots panel fills the campaign's declared slots.

## Success criteria

1. With `VOX_CAMPAIGN=bharat_matka` (the default), the console and telephony bridges run the Anaaya script and its declared slots; the opening greeting is Anaaya's.
2. `build_voicebot_system_prompt` is provably campaign-agnostic: a unit test with a non-Bharat-Matka fixture shows the prompt embeds whatever `objective`/`knowledge`/`dos`/`donts`/slots the script declares, plus the fixed customer-led policy text. No campaign-specific literal exists in `prompts.py` or the loader.
3. Slots are campaign-driven: the agent is built with the campaign's `SlotSchema`, and a different campaign file with different slots would yield a different schema with no code change.
4. Existing callers and tests are unaffected (new `VoiceBotScript` fields are optional; the new factory `slots=` param defaults to `SlotSchema()`; `DEFAULT_DEMO_SCRIPT` still valid).
5. A missing campaign file degrades gracefully to the demo script + empty slots with a logged warning.
