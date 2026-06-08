"""System prompt builder for VoiceBot + ChatBot.

Builds the system message that goes to the LLM. The prompt:
- Identifies the agent (name, role, company)
- Sets the language and code-switching policy
- Embeds the talking points / qualifying questions / objection responses
- Lists the slots the agent should try to fill
- Specifies the structured JSON response schema (PRD §12.2 / §12.3)

Kept as a pure-Python builder rather than a templating engine so it's easy
to inspect, diff, and unit-test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from src.dialogue.slots import SlotSchema


class _SafeDict(dict):
    """dict for str.format_map that leaves unknown ``{tokens}`` intact."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_opening(script: "VoiceBotScript", lead_data: dict[str, Any]) -> str:
    """Substitute known template tokens in the opening for the prompt context.

    Mirrors the tokens the telephony layer renders for the spoken opening
    ({agent_name}, {lead_name}, company_name, plus any lead_data keys).
    Unknown tokens are left as-is so a bad template never raises.
    """
    variables = {
        "agent_name": script.agent_name,
        "company_name": script.company_name,
        "lead_name": (lead_data or {}).get("lead_name", "ji"),
        **(lead_data or {}),
    }
    try:
        return script.opening.strip().format_map(_SafeDict(variables))
    except Exception:
        return script.opening.strip()


VOICEBOT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response_text", "language", "action"],
    "properties": {
        "response_text": {"type": "string"},
        "language": {"type": "string"},
        "conversation_phase": {
            "type": "string",
            "enum": ["opening", "pitch", "qualification", "objection", "closing"],
        },
        "updated_slots": {"type": "object"},
        "action": {
            "type": "string",
            "enum": [
                "continue",
                "clarify",
                "transfer",
                "schedule_callback",
                "send_info",
                "close_positive",
                "close_negative",
                "end",
            ],
        },
        "action_reason": {"type": "string"},
        "sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "negative", "frustrated"],
        },
        "internal_notes": {"type": "string"},
    },
}

CHATBOT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["response_text", "language"],
    "properties": {
        "response_text": {"type": "string"},
        "language": {"type": "string"},
        "sources_used": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "action": {
            "type": "string",
            "enum": ["none", "schedule_callback", "send_info", "create_ticket", "escalate"],
        },
        "suggested_followups": {"type": "array", "items": {"type": "string"}},
    },
}


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

    # Language policy. The reply is spoken by an Indic (e.g. Hindi) TTS that
    # cannot pronounce Latin script, so response_text MUST be in the native
    # script — romanized/English text comes out garbled ("drunk").
    parts.append(
        f"Speak in {script.language_default}. CRITICAL: write `response_text` ONLY in the "
        "native script of that language (Devanagari for Hindi) — NEVER in romanized/Latin "
        "letters or English sentences. Your reply is spoken aloud by a Hindi text-to-speech "
        "voice that mispronounces Latin script. So even if the user speaks in English or "
        "types Hinglish in Latin letters, you reply in natural, warm Hindi written in "
        "Devanagari. Match their level of formality. (Well-known brand names may stay as-is.)"
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
        parts.append(
            "Opening line (already spoken at the start of the call):\n"
            + _render_opening(script, lead_data)
        )

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
        "- Begin every reply with a SHORT first sentence (about 2-5 words) — a brief, "
        "natural acknowledgment that ends in a sentence boundary — and THEN continue "
        "with any detail or question. This lets your voice start playing sooner. "
        "VARY this opener across turns so it never sounds scripted (e.g. rotate among "
        "जी, अच्छा, बिल्कुल, समझ गई, हाँ, ठीक है). Skip the opener only when the whole "
        "reply is already that short.\n"
        "- Never invent facts about the company or its products.\n"
        "- If the user asks if you are AI, answer honestly.\n"
        "- If the user asks to be removed, set action=close_negative and acknowledge.\n"
        "- Callback: do NOT set action=schedule_callback until you have a SPECIFIC "
        "day and time. If the user is vague ('kal', 'baad mein', 'later'), keep "
        "action=continue, ask for the exact time (e.g. 'Kal kis samay call karoon?'), "
        "and save it in updated_slots.callback_time. Only schedule_callback once a "
        "concrete time is confirmed.\n"
        "- Set action=end only when the conversation is genuinely over."
    )

    if extra_directives:
        parts.append("Additional directives:\n" + "\n".join(f"- {d}" for d in extra_directives))

    return "\n\n".join(parts)


def build_chatbot_system_prompt(
    company_name: str,
    language_default: str = "en",
    rag_context: Optional[str] = None,
    extra_directives: Optional[list[str]] = None,
) -> str:
    """System prompt for the RAG-powered ChatBot agent (Phase 4)."""
    parts: list[str] = []
    parts.append(
        f"You are a helpful assistant for {company_name}. Answer the user's question "
        "using only the provided sources. If the sources don't contain the answer, "
        "say you don't know — do not invent."
    )
    parts.append(f"Default language: {language_default}. Mirror the user's language.")

    if rag_context:
        parts.append("Reference sources:\n" + rag_context)

    parts.append(
        "Respond with a single JSON object matching this schema:\n"
        + json.dumps(CHATBOT_RESPONSE_SCHEMA, indent=2)
    )

    if extra_directives:
        parts.append("Additional directives:\n" + "\n".join(f"- {d}" for d in extra_directives))

    return "\n\n".join(parts)
