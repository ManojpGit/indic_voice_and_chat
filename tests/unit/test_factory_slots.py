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
