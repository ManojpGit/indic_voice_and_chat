# Customer-Led Agent + Campaign-Upfront Script Loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load one campaign (script + declared slots) at startup from config, and make the agent customer-led (answer the customer's actual question first, then advance the campaign objective, redirect only when off-topic) — with all logic campaign-agnostic.

**Architecture:** Extend `VoiceBotScript` for richer optional fields and rewrite the system-prompt builder to be generic over whatever a script declares. A new `campaign_loader` reads a campaign YAML into `LoadedCampaign(script, slots)` (slots via the existing `SlotSchema.from_campaign_yaml`). The three bridge factories gain a `slots=` param; `main.py` loads the active campaign once and passes script + slots into them. Per-turn JSON contract and state machine are unchanged.

**Tech Stack:** Python, dataclasses, PyYAML (already a dep), pytest. No new third-party deps.

**Spec:** `docs/superpowers/specs/2026-06-02-customer-led-agent-campaign-script-design.md`

---

## Reference: existing code this plan touches

- `src/dialogue/prompts.py`: `VoiceBotScript` (dataclass + `from_campaign_yaml`) and `build_voicebot_system_prompt(script, schema, lead_data=None, extra_directives=None)` and `VOICEBOT_RESPONSE_SCHEMA`.
- `src/dialogue/slots.py`: `SlotSchema.from_campaign_yaml(slots_dict) -> SlotSchema` (already exists; maps each entry to a `SlotSpec` with `type`/`required`/`values`). `SlotFiller(schema)` exposes `.schema` and `.values`.
- `src/bootstrap.py`: `DEFAULT_DEMO_SCRIPT`; `make_bridge_factory(providers, session_store=None, bridge_config=None, script=DEFAULT_DEMO_SCRIPT)` and `make_exotel_bridge_factory(...)`. Both build `VoiceBotAgent(..., slot_schema=SlotSchema(), script=script, ...)` (lines ~166-169 and ~283-286).
- `src/api/dev_console.py`: `make_browser_bridge_factory(providers, script=DEFAULT_DEMO_SCRIPT)` builds `VoiceBotAgent(..., slot_schema=SlotSchema(), ...)` (lines ~123-126).
- `src/main.py` lifespan: registers the three factories with `providers=providers` (lines ~107, ~110, ~113).
- `tests/unit/test_prompts.py`: asserts the prompt contains agent name, company, talking points, qualifying question, the `is_ai` objection key, lead data, `* lead_name`/`* interest_level` slot markers, the JSON schema fields, and appended `extra_directives`. **The Task 2 rewrite must keep all of these passing.**

Run everything with the venv python: `.venv/bin/python -m pytest ...`.

---

## Task 1: Extend `VoiceBotScript` + `from_campaign_yaml`

**Files:**
- Modify: `src/dialogue/prompts.py` (the `VoiceBotScript` dataclass + `from_campaign_yaml`)
- Test: `tests/unit/test_prompts.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/unit/test_prompts.py`:

```python
def test_from_campaign_yaml_parses_new_fields_and_aliases() -> None:
    s = VoiceBotScript.from_campaign_yaml({
        "name": "Anaaya", "company": "Bharat Matka", "role": "Sales",
        "language": "hi", "greeting": "Namaste",
        "objective": "Push link", "knowledge": {"safety": "It is safe"},
        "dos": ["Be warm"], "donts": ["No jargon"],
        "personality": "warm", "gender": "female",
        "conversation_style": "Hinglish", "max_turns": 12,
        "closing": "Dhanyavaad!",   # a string, not a dict
    })
    assert s.agent_name == "Anaaya"
    assert s.company_name == "Bharat Matka"
    assert s.agent_role == "Sales"
    assert s.language_default == "hi"
    assert s.opening == "Namaste"
    assert s.objective == "Push link"
    assert s.knowledge == {"safety": "It is safe"}
    assert s.dos == ["Be warm"] and s.donts == ["No jargon"]
    assert s.personality == "warm" and s.gender == "female"
    assert s.conversation_style == "Hinglish" and s.max_turns == 12
    assert s.closing == {"default": "Dhanyavaad!"}   # string normalized to dict


def test_from_campaign_yaml_backcompat_existing_keys() -> None:
    s = VoiceBotScript.from_campaign_yaml({
        "agent_name": "P", "agent_role": "R", "company_name": "C",
        "closing": {"positive": "ok"},
    })
    assert s.agent_name == "P" and s.closing == {"positive": "ok"}
    assert s.knowledge == {} and s.max_turns == 0 and s.dos == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_prompts.py::test_from_campaign_yaml_parses_new_fields_and_aliases -v`
Expected: FAIL — `TypeError`/`AttributeError` (new fields/kwargs don't exist yet).

- [ ] **Step 3: Replace the `VoiceBotScript` dataclass and `from_campaign_yaml`** in `src/dialogue/prompts.py` with:

```python
@dataclass
class VoiceBotScript:
    agent_name: str
    agent_role: str
    company_name: str
    language_default: str = "hi"
    opening: str = ""
    talking_points: list[str] = field(default_factory=list)
    qualifying_questions: list[str] = field(default_factory=list)
    objection_responses: dict[str, str] = field(default_factory=dict)
    closing: dict[str, str] = field(default_factory=dict)
    # Richer, optional campaign fields. All default empty so existing callers
    # and DEFAULT_DEMO_SCRIPT are unaffected. The prompt builder consumes
    # whatever these contain — no campaign-specific assumptions live in code.
    personality: str = ""
    gender: str = ""
    objective: str = ""
    knowledge: dict[str, str] = field(default_factory=dict)
    dos: list[str] = field(default_factory=list)
    donts: list[str] = field(default_factory=list)
    conversation_style: str = ""
    max_turns: int = 0

    @classmethod
    def from_campaign_yaml(cls, script: dict[str, Any]) -> "VoiceBotScript":
        def pick(*keys: str, default: str = "") -> str:
            for k in keys:
                if script.get(k) is not None:
                    return script[k]
            return default

        closing_raw = script.get("closing")
        if isinstance(closing_raw, str):
            closing = {"default": closing_raw}
        else:
            closing = dict(closing_raw or {})

        return cls(
            agent_name=pick("agent_name", "name", default="Agent"),
            agent_role=pick("agent_role", "role", default="Customer Engagement"),
            company_name=pick("company_name", "company", default="[Company]"),
            language_default=pick("language_default", "language", default="hi"),
            opening=pick("opening", "greeting", default=""),
            talking_points=list(script.get("talking_points") or []),
            qualifying_questions=list(script.get("qualifying_questions") or []),
            objection_responses=dict(script.get("objection_responses") or {}),
            closing=closing,
            personality=script.get("personality", "") or "",
            gender=script.get("gender", "") or "",
            objective=script.get("objective", "") or "",
            knowledge=dict(script.get("knowledge") or {}),
            dos=list(script.get("dos") or []),
            donts=list(script.get("donts") or []),
            conversation_style=script.get("conversation_style", "") or "",
            max_turns=int(script.get("max_turns") or 0),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_prompts.py -v`
Expected: PASS (the two new tests + all existing prompt tests still green — the new fields are additive).

- [ ] **Step 5: Commit**

```bash
git add src/dialogue/prompts.py tests/unit/test_prompts.py
git commit -m "extend VoiceBotScript with campaign fields + key-alias/string-closing parsing"
```

---

## Task 2: Rewrite `build_voicebot_system_prompt` to be customer-led + generic

**Files:**
- Modify: `src/dialogue/prompts.py` (`build_voicebot_system_prompt`)
- Test: `tests/unit/test_prompts.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/unit/test_prompts.py`:

```python
def test_voicebot_prompt_is_generic_over_script() -> None:
    """The builder must embed whatever the script declares — no hardcoded
    campaign content. Uses sentinel strings (not Bharat Matka)."""
    script = VoiceBotScript.from_campaign_yaml({
        "agent_name": "Zeta", "agent_role": "Helper", "company_name": "Foo Inc",
        "objective": "SENTINEL_OBJECTIVE_X",
        "knowledge": {"q1": "SENTINEL_KNOWLEDGE_Y"},
        "dos": ["SENTINEL_DO_Z"],
        "donts": ["SENTINEL_DONT_W"],
        "personality": "SENTINEL_PERSONA",
        "max_turns": 7,
    })
    prompt = build_voicebot_system_prompt(script, SlotSchema())
    for sentinel in ("SENTINEL_OBJECTIVE_X", "SENTINEL_KNOWLEDGE_Y",
                     "SENTINEL_DO_Z", "SENTINEL_DONT_W", "SENTINEL_PERSONA"):
        assert sentinel in prompt
    # Fixed customer-led policy text is present regardless of campaign.
    assert "LISTEN FIRST" in prompt
    assert "REDIRECT ONLY WHEN" in prompt
    # Soft turn budget surfaced from the script's max_turns.
    assert "7 turns" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_prompts.py::test_voicebot_prompt_is_generic_over_script -v`
Expected: FAIL — `assert "SENTINEL_OBJECTIVE_X" in prompt` (the current builder ignores `objective`/`knowledge`/`dos`/`donts`).

- [ ] **Step 3: Replace `build_voicebot_system_prompt`** in `src/dialogue/prompts.py` with this generic, customer-led version (it preserves every section the existing tests assert and adds the new ones):

```python
def build_voicebot_system_prompt(
    script: VoiceBotScript,
    schema: SlotSchema,
    lead_data: Optional[dict[str, Any]] = None,
    extra_directives: Optional[list[str]] = None,
) -> str:
    """Assemble the VoiceBotAgent system prompt.

    Campaign-agnostic: this builder only embeds what ``script`` and ``schema``
    declare. The customer-led policy is fixed (applies to every campaign);
    all campaign-specific content comes from the script fields.
    """
    lead_data = lead_data or {}
    parts: list[str] = []

    # Identity + persona.
    parts.append(
        f"You are {script.agent_name}, a {script.agent_role} at {script.company_name}. "
        f"You are on a phone call with a lead. Speak naturally as a human would on a call."
    )
    if script.personality:
        parts.append(f"Your personality: {script.personality}.")
    if script.conversation_style:
        parts.append(f"Conversation style: {script.conversation_style}.")

    # Language policy.
    parts.append(
        f"Default language: {script.language_default}. Mirror the user's language and "
        "match their level of formality. Code-switch (Hindi/English mixing) is fine "
        "if the user does it."
    )

    # Customer-led behavior (fixed policy, generic over every campaign).
    parts.append(
        "How to handle every turn — this is your core behavior:\n"
        "1. LISTEN FIRST. Work out what the customer actually said, then answer THAT "
        "directly and helpfully before anything else. Draw on the knowledge below, in "
        "your own warm words — never recite.\n"
        "2. THEN gently move toward your objective. The talking points are material to "
        "draw on, not a checklist to read out.\n"
        "3. REDIRECT ONLY WHEN the customer's input is totally unrelated to this call "
        "(e.g. weather, wrong number, personal chit-chat): briefly and warmly acknowledge, "
        "then steer back. If their question is on-topic or a concern, answer it — never deflect.\n"
        "4. Follow the do's and don'ts below for tone."
    )

    if script.objective:
        parts.append("Your objective on this call:\n" + script.objective.strip())

    if script.opening:
        parts.append("Opening line (already spoken at the start of the call):\n" + script.opening.strip())

    if script.talking_points:
        bullets = "\n".join(f"- {p}" for p in script.talking_points)
        parts.append("Talking points (material, not a checklist):\n" + bullets)

    if script.qualifying_questions:
        bullets = "\n".join(f"- {q}" for q in script.qualifying_questions)
        parts.append("Qualifying questions to ask when natural:\n" + bullets)

    # Merge the campaign's knowledge base and objection responses into one
    # reference set the agent uses to answer questions/concerns.
    knowledge_items = {**(script.knowledge or {}), **(script.objection_responses or {})}
    if knowledge_items:
        bullets = "\n".join(f"- {tag}: {resp}" for tag, resp in knowledge_items.items())
        parts.append(
            "Knowledge for answering the customer's questions and concerns (use the "
            "substance in your own words, not verbatim):\n" + bullets
        )

    if script.closing:
        bullets = "\n".join(f"- {tag}: {resp}" for tag, resp in script.closing.items())
        parts.append("Closing lines:\n" + bullets)

    if script.dos:
        parts.append("Do:\n" + "\n".join(f"- {d}" for d in script.dos))
    if script.donts:
        parts.append("Don't:\n" + "\n".join(f"- {d}" for d in script.donts))

    if script.max_turns and script.max_turns > 0:
        parts.append(
            f"You have roughly {script.max_turns} turns. If the customer clearly is not "
            "engaging after a few honest attempts, close gracefully rather than pushing."
        )

    if schema.specs:
        slot_lines = []
        for name, spec in schema.specs.items():
            mark = "*" if spec.required else " "
            extra = (
                f" (one of: {', '.join(spec.values)})"
                if spec.values
                else f" ({spec.type.value})"
            )
            slot_lines.append(f"  {mark} {name}{extra}")
        parts.append(
            "Slots to fill (* = required). Update them via the JSON `updated_slots` field "
            "as you learn from the user:\n" + "\n".join(slot_lines)
        )

    if lead_data:
        parts.append("Known lead data:\n" + json.dumps(lead_data, ensure_ascii=False, indent=2))

    parts.append(
        "On every turn you MUST respond with a single JSON object matching this schema:\n"
        + json.dumps(VOICEBOT_RESPONSE_SCHEMA, indent=2)
    )

    parts.append(
        "Rules:\n"
        "- Keep `response_text` concise (1-2 sentences) — this is voice, not chat.\n"
        "- Never invent facts about the company or its products.\n"
        "- If the user asks if you are AI, answer honestly.\n"
        "- If the user asks to be removed, set action=close_negative and acknowledge.\n"
        "- Set action=end only when the conversation is genuinely over."
    )

    if extra_directives:
        parts.append("Additional directives:\n" + "\n".join(f"- {d}" for d in extra_directives))

    return "\n\n".join(parts)
```

- [ ] **Step 4: Run the prompt tests + full unit suite**

Run: `.venv/bin/python -m pytest tests/unit/test_prompts.py -v`
Expected: PASS (new generic test + all existing prompt assertions — agent name, company, talking points, qualifying question, `is_ai`, lead data, slot markers, schema fields, extra directives — still hold).

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: PASS. If `tests/unit/test_voicebot_agent.py` asserts any old prompt wording that changed, update only that assertion to the new wording (the agent behavior is unchanged; only prompt text moved). Re-run until green.

- [ ] **Step 5: Commit**

```bash
git add src/dialogue/prompts.py tests/unit/test_prompts.py
git commit -m "prompts: customer-led, campaign-agnostic system prompt builder"
```

---

## Task 3: Create the campaign loader

**Files:**
- Create: `src/dialogue/campaign_loader.py`
- Test: `tests/unit/test_campaign_loader.py`

- [ ] **Step 1: Write the failing test** — create `tests/unit/test_campaign_loader.py`:

```python
from __future__ import annotations

import textwrap
from pathlib import Path

from src.dialogue.campaign_loader import (
    LoadedCampaign,
    active_campaign_slug,
    load_campaign,
)
from src.dialogue.slots import SlotSchema


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / f"{name}.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_campaign_reads_script_and_slots(tmp_path: Path) -> None:
    _write(tmp_path, "demo", """
        campaign:
          agent: { name: Zed, company: ZCo, role: Sales, language: hi }
          script:
            greeting: "Hi"
            objective: "Do X"
            knowledge: { safety: "safe" }
          slots:
            interest_level: { type: enum, required: true, values: [hot, cold] }
            wants_link: { type: boolean }
    """)
    lc = load_campaign("demo", campaigns_dir=tmp_path)
    assert isinstance(lc, LoadedCampaign)
    assert lc.script.agent_name == "Zed"
    assert lc.script.company_name == "ZCo"
    assert lc.script.opening == "Hi"
    assert lc.script.objective == "Do X"
    assert lc.script.knowledge == {"safety": "safe"}
    assert "interest_level" in lc.slots.specs
    assert lc.slots.specs["interest_level"].required is True
    assert lc.slots.specs["interest_level"].values == ["hot", "cold"]
    assert "wants_link" in lc.slots.specs


def test_load_campaign_missing_slug_falls_back(tmp_path: Path) -> None:
    lc = load_campaign("does_not_exist", campaigns_dir=tmp_path)
    assert lc.script.agent_name  # DEFAULT_DEMO_SCRIPT is populated
    assert lc.slots.specs == {}  # empty schema on fallback


def test_active_campaign_slug_default_and_env(monkeypatch) -> None:
    monkeypatch.delenv("VOX_CAMPAIGN", raising=False)
    assert active_campaign_slug() == "bharat_matka"
    monkeypatch.setenv("VOX_CAMPAIGN", "foo")
    assert active_campaign_slug() == "foo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_campaign_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.dialogue.campaign_loader'`.

- [ ] **Step 3: Create `src/dialogue/campaign_loader.py`**:

```python
"""Campaign-upfront script loading.

Reads one campaign YAML (selected by VOX_CAMPAIGN) into a script + slot schema
at startup. The active campaign drives every call this process handles. The
loader is campaign-agnostic — it only parses whatever the YAML declares.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.dialogue.prompts import VoiceBotScript
from src.dialogue.slots import SlotSchema

log = logging.getLogger(__name__)

DEFAULT_CAMPAIGN_SLUG = "bharat_matka"
DEFAULT_CAMPAIGNS_DIR = Path("config/campaigns")


@dataclass
class LoadedCampaign:
    script: VoiceBotScript
    slots: SlotSchema


def active_campaign_slug() -> str:
    """The campaign slug to load at startup (env VOX_CAMPAIGN, default bharat_matka)."""
    return os.environ.get("VOX_CAMPAIGN", DEFAULT_CAMPAIGN_SLUG)


def load_campaign(
    slug: str, campaigns_dir: Path = DEFAULT_CAMPAIGNS_DIR
) -> LoadedCampaign:
    """Load ``config/campaigns/<slug>.yaml`` into a script + slot schema.

    Missing/unreadable file -> warn and fall back to the demo script with an
    empty slot schema, so the app still boots.
    """
    path = campaigns_dir / f"{slug}.yaml"
    if not path.exists():
        from src.bootstrap import DEFAULT_DEMO_SCRIPT  # lazy: avoid import cost/cycle

        log.warning("campaign file not found: %s; using demo script", path)
        return LoadedCampaign(DEFAULT_DEMO_SCRIPT, SlotSchema())

    with path.open() as f:
        data = yaml.safe_load(f) or {}
    camp = data.get("campaign", data)  # tolerate with/without the wrapper
    merged = {**(camp.get("agent") or {}), **(camp.get("script") or {})}
    script = VoiceBotScript.from_campaign_yaml(merged)
    slots = SlotSchema.from_campaign_yaml(camp.get("slots") or {})
    return LoadedCampaign(script, slots)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_campaign_loader.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dialogue/campaign_loader.py tests/unit/test_campaign_loader.py
git commit -m "add campaign loader: LoadedCampaign(script, slots) from campaign YAML"
```

---

## Task 4: Author the Bharat Matka / Anaaya campaign file

**Files:**
- Create: `config/campaigns/bharat_matka.yaml`
- Test: `tests/unit/test_campaign_loader.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/unit/test_campaign_loader.py`:

```python
def test_bharat_matka_campaign_loads_from_default_dir() -> None:
    # Uses the real config/campaigns/bharat_matka.yaml (default dir).
    lc = load_campaign("bharat_matka")
    assert lc.script.agent_name == "Anaaya"
    assert lc.script.company_name == "Bharat Matka"
    assert lc.script.gender == "female"
    assert lc.script.knowledge  # non-empty knowledge base
    assert lc.script.dos and lc.script.donts
    assert lc.script.max_turns == 12
    # Declared slots are present (campaign-driven; names are sample data).
    assert lc.slots.specs
    assert "interest_level" in lc.slots.specs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_campaign_loader.py::test_bharat_matka_campaign_loads_from_default_dir -v`
Expected: FAIL — fallback to demo script (file missing), so `agent_name != "Anaaya"`.

- [ ] **Step 3: Create `config/campaigns/bharat_matka.yaml`** with exactly this content:

```yaml
campaign:
  id: bharat_matka
  name: "Bharat Matka Official App"
  status: active

  agent:
    name: "Anaaya"
    company: "Bharat Matka"
    role: "Support & Sales"
    personality: "warm, energetic, and street-smart helpful"
    language: "hi"
    gender: "female"

  script:
    greeting: "हेलो, नमस्ते सर! मैं {agent_name} बात कर रही हूं भारत मटका Official App से। क्या मेरी बात {lead_name} जी से हो रही है?"

    objective: |
      Establish trust immediately, mention the 'Official' status, and push for the
      WhatsApp link sharing by highlighting the bonus.

    talking_points:
      - "Aap ko ek choti si jankari dene ke liye call kiya hai — kya aapke paas ek minute hai?"
      - "Ye market ka sabse TRUSTED aur SAFE application hai — Matka, Aviator, Casino sab ek jagah"
      - "Withdrawal bilkul INSTANT hai — jeeta hua paisa turant aapke account mein"
      - "Abhi join karne par REGISTRATION FREE hai, pehle deposit par 10% CASH BONUS, sirf 100 rupay se start"
      - "Doston ko refer karne par 3% COMMISSION milega har Monday"
      - "24x7 WhatsApp support — koi bhi problem aaye toh turant help"
      - "WhatsApp par App ka link aur Bonus details share kar doon? Aap check kar lena"

    knowledge:
      safety: "Sir, main samajh sakti hun aapka concern. Par ye OFFICIAL Bharat Matka app hai. Hazaron log roz khel rahe hain. Aap befikar rahiye."
      scam_concerns: "Sir, trust kijiye, main aapko PERSONALLY GUIDE karungi. Aap chote amount se, bas 100 rupay se check karke dekhiye. Withdrawal turant milega."
      withdrawal: "Sir, withdrawal ki koi tension nahi. 500 se lekar jitna marzi nikal lijiye, sab instant hai."
      deposit: "Sir, Minimum Deposit bas 100 rupay hai. Aap try karne ke liye chota amount daal sakte hain."
      transaction_speed: "Sir, Deposit aur Withdrawal dono AUTOMATIC aur FAST hain. 5 minute mein paisa aa jata hai."
      support: "Arey Sir, main hu na. Koi bhi dikkat aayi toh hamara WhatsApp Support 24 ghante on rehta hai. Turant reply milega."
      referral: "Doston ko refer karne par 3% COMMISSION milega har Monday. Khelne ka bhi paisa, refer karne ka bhi."

    closing: "Ok Sir, Thank you time dene ke liye. Aapka din shubh rahe!"

    dos:
      - "Keep tone 'High Energy' but polite"
      - "Use filler words like 'Actually', 'Dekhiye', 'Bas' to sound human"
      - "Always end turns with a question or a prompt (e.g., 'Sahi hai?', 'Bhej du?')"

    donts:
      - "Don't sound robotic or monotone"
      - "Don't use complex Hindi words like 'Jankari' (Use 'Info' or 'Update')"
      - "Don't argue if they say it's a scam — reassure them calmly"
      - "Never forget to mention the '10% Bonus' — that is the hook"

    max_turns: 12
    conversation_style: "Colloquial Hinglish, Warm, Persuasive"

  # Slots are campaign-declared sample data — edit freely per campaign.
  slots:
    interest_level:
      type: enum
      required: true
      values: [hot, warm, cold]
    wants_whatsapp_link:
      type: boolean
    deposit_intent:
      type: boolean
    main_objection:
      type: enum
      values: [scam, safety, withdrawal, deposit, none]
    callback_time:
      type: datetime
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_campaign_loader.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add config/campaigns/bharat_matka.yaml tests/unit/test_campaign_loader.py
git commit -m "add Bharat Matka / Anaaya campaign (script + declared slots)"
```

---

## Task 5: Thread per-campaign slots through the three bridge factories

**Files:**
- Modify: `src/bootstrap.py` (`make_bridge_factory`, `make_exotel_bridge_factory`)
- Modify: `src/api/dev_console.py` (`make_browser_bridge_factory`)
- Test: `tests/unit/test_factory_slots.py`

- [ ] **Step 1: Write the failing test** — create `tests/unit/test_factory_slots.py`:

```python
from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

from src.api.dev_console import make_browser_bridge_factory
from src.bootstrap import make_bridge_factory, make_exotel_bridge_factory
from src.dialogue.slots import SlotSchema


def test_all_factories_accept_slots_param_defaulting_empty() -> None:
    for fn in (make_bridge_factory, make_exotel_bridge_factory, make_browser_bridge_factory):
        params = inspect.signature(fn).parameters
        assert "slots" in params, f"{fn.__name__} missing slots param"
        default = params["slots"].default
        assert isinstance(default, SlotSchema) and default.specs == {}, fn.__name__


def test_browser_factory_passes_slots_into_agent() -> None:
    slots = SlotSchema.from_campaign_yaml({"foo": {"type": "string"}})
    providers = SimpleNamespace(
        get_stt=lambda t: Mock(), get_llm=lambda t: Mock(), get_tts=lambda t: Mock(),
    )
    pipeline = SimpleNamespace(
        stt=SimpleNamespace(language="hi-IN"),
        llm=SimpleNamespace(temperature=0.5, max_tokens=256, response_format="json"),
        tts=SimpleNamespace(language="hi-IN", voice_id=None),
    )
    tenant = SimpleNamespace(slug="dev", id="t1", settings=SimpleNamespace(pipeline=pipeline))
    factory = make_browser_bridge_factory(providers, slots=slots)
    bridge = factory(websocket=object(), tenant=tenant)
    # The agent's slot filler must hold the campaign's schema, not an empty one.
    assert bridge._agent.slots.schema is slots
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_factory_slots.py -v`
Expected: FAIL — `assert "slots" in params` (factories don't take `slots` yet).

- [ ] **Step 3a: Edit `src/bootstrap.py`** — `make_bridge_factory`. Change the signature:

```python
def make_bridge_factory(
    providers: TenantProviders,
    session_store: SessionStore | None = None,
    bridge_config: TwilioBridgeConfig | None = None,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
) -> Callable[[WebSocket, TenantContext], TwilioMediaBridge]:
```

and in that factory's `VoiceBotAgent(...)` construction change `slot_schema=SlotSchema(),` to `slot_schema=slots,`.

- [ ] **Step 3b: Edit `src/bootstrap.py`** — `make_exotel_bridge_factory`. Same two changes: add `slots: SlotSchema = SlotSchema(),` to the signature (after `script`), and change its `VoiceBotAgent(...)` `slot_schema=SlotSchema(),` to `slot_schema=slots,`.

- [ ] **Step 3c: Edit `src/api/dev_console.py`** — `make_browser_bridge_factory`. Change the signature:

```python
def make_browser_bridge_factory(
    providers: TenantProviders,
    script: VoiceBotScript = DEFAULT_DEMO_SCRIPT,
    slots: SlotSchema = SlotSchema(),
) -> BrowserBridgeFactory:
```

and change its `VoiceBotAgent(...)` `slot_schema=SlotSchema(),` to `slot_schema=slots,`.

Note: `SlotSchema` is already imported in both files. A shared default `SlotSchema()` instance is safe here because `SlotFiller` copies values and never mutates the schema.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_factory_slots.py -v`
Expected: PASS (2 passed). If `test_browser_factory_passes_slots_into_agent` errors inside `PipelineEngine`/`VoiceBotAgent` construction, the stubs are fine — `Mock()` providers and the `SimpleNamespace` tenant satisfy every attribute the factory reads; do not weaken the `is slots` assertion.

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: PASS (full suite green).

- [ ] **Step 5: Commit**

```bash
git add src/bootstrap.py src/api/dev_console.py tests/unit/test_factory_slots.py
git commit -m "factories: accept per-campaign slot schema and pass it to the agent"
```

---

## Task 6: Load the active campaign in `main.py` and pass script + slots to the factories

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Add the import** near the other `src.*` imports in `src/main.py`:

```python
from src.dialogue.campaign_loader import active_campaign_slug, load_campaign
```

- [ ] **Step 2: Load the campaign once** in `lifespan`, immediately AFTER the `providers = build_provider_registry(...)` block and BEFORE the `telephony_hooks.set_bridge_factory(...)` call:

```python
    campaign = load_campaign(active_campaign_slug())
    log.info(
        "campaign loaded",
        extra={"slug": active_campaign_slug(), "agent": campaign.script.agent_name,
               "slots": list(campaign.slots.specs.keys())},
    )
```

- [ ] **Step 3: Pass `script` + `slots` into all three factory calls.** Replace the existing three registrations:

```python
    telephony_hooks.set_bridge_factory(
        make_bridge_factory(providers=providers, session_store=base_session_store)
    )
    telephony_hooks.set_exotel_bridge_factory(
        make_exotel_bridge_factory(providers=providers, session_store=base_session_store)
    )
    ...
    if dev_console_enabled():
        set_browser_bridge_factory(make_browser_bridge_factory(providers=providers))
```

with:

```python
    telephony_hooks.set_bridge_factory(
        make_bridge_factory(
            providers=providers, session_store=base_session_store,
            script=campaign.script, slots=campaign.slots,
        )
    )
    telephony_hooks.set_exotel_bridge_factory(
        make_exotel_bridge_factory(
            providers=providers, session_store=base_session_store,
            script=campaign.script, slots=campaign.slots,
        )
    )
    ...
    if dev_console_enabled():
        set_browser_bridge_factory(
            make_browser_bridge_factory(
                providers=providers, script=campaign.script, slots=campaign.slots,
            )
        )
```

(Leave the rest of the `dev_console_enabled()` block — the log line — as is.)

- [ ] **Step 4: Verify the app imports cleanly**

Run: `VOX_DEV_CONSOLE=1 .venv/bin/python -c "import src.main; print('ok')"`
Expected: prints `ok`.

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: PASS (full suite green).

- [ ] **Step 5: Commit**

```bash
git add src/main.py
git commit -m "main: load active campaign at startup and feed script+slots to bridges"
```

---

## Task 7: Full suite + manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full unit suite**

Run: `.venv/bin/python -m pytest tests/unit/ -q`
Expected: PASS (all green, including the new prompt, campaign-loader, and factory-slots tests).

- [ ] **Step 2: Confirm the active campaign resolves to Anaaya**

Run: `.venv/bin/python -c "from src.dialogue.campaign_loader import load_campaign, active_campaign_slug; lc = load_campaign(active_campaign_slug()); print(lc.script.agent_name, '|', list(lc.slots.specs.keys()))"`
Expected: prints `Anaaya | ['interest_level', 'wants_whatsapp_link', 'deposit_intent', 'main_objection', 'callback_time']`

- [ ] **Step 3: Launch the console and verify the greeting + prompt**

Run (background): `VOX_DEV_CONSOLE=1 VOX_CAMPAIGN=bharat_matka .venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8765 --env-file .env`
Then: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/dev/voice`
Expected: `200`, and the startup log shows `campaign loaded` with `agent=Anaaya`.

- [ ] **Step 4: Manual browser check (human, with headphones)**

Open `http://localhost:8765/dev/voice`, click Start.
Expected: Anaaya's Hindi greeting plays and shows in the transcript; the state/slots panel shows the campaign's declared slots. Ask an on-topic question (e.g. "withdrawal kaise hota hai?") — she answers it first (from `knowledge`), then nudges toward the objective. Say something totally unrelated (e.g. about the weather) — she briefly acknowledges and steers back. Slots like `interest_level` / `wants_whatsapp_link` update as the conversation warrants.

- [ ] **Step 5: Final commit (empty allowed if no fixes were needed)**

```bash
git add -A
git commit -m "verify customer-led Anaaya campaign end-to-end" --allow-empty
```

---

## Notes for the implementer

- **Run everything with `.venv/bin/python`** (not the conda base).
- **Keep it campaign-agnostic.** No campaign-specific strings ("10% bonus", "WhatsApp", slot names) belong in `prompts.py`, `campaign_loader.py`, or the factories — only in `config/campaigns/*.yaml` and test fixtures.
- **The prompt rewrite must keep the existing `test_prompts.py` assertions passing** (agent name, company, talking points, qualifying question, `is_ai`, lead data, `* lead_name`/`* interest_level`, schema fields, extra directives). The provided builder preserves every one of those sections.
- The dev console's slots panel was empty before because the factory hardcoded `SlotSchema()`. After Task 5/6 it shows the campaign's declared slots — that's the visible proof slots are now campaign-driven.
