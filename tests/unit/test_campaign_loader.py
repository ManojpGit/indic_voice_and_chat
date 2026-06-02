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
