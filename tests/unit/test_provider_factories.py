from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.providers import (
    UnknownProviderError,
    get_llm_provider,
    get_stt_provider,
    get_telephony_provider,
    get_tts_provider,
    get_vector_store,
)
from src.providers.llm.groq import GroqLLMAdapter
from src.providers.stt.sarvam import SarvamSTTAdapter
from src.providers.telephony.twilio import TwilioAdapter
from src.providers.tts.sarvam import SarvamTTSAdapter
from src.providers.vector_store.faiss_store import FAISSAdapter


def test_stt_factory_returns_sarvam() -> None:
    assert isinstance(
        get_stt_provider({"provider": "sarvam", "api_key": "x"}), SarvamSTTAdapter
    )


def test_llm_factory_returns_groq() -> None:
    assert isinstance(
        get_llm_provider({"provider": "groq", "client": MagicMock()}), GroqLLMAdapter
    )


def test_tts_factory_returns_sarvam() -> None:
    assert isinstance(
        get_tts_provider({"provider": "sarvam", "api_key": "x"}), SarvamTTSAdapter
    )


def test_telephony_factory_returns_twilio() -> None:
    inst = get_telephony_provider(
        {"provider": "twilio", "client": MagicMock(), "account_sid": "ACx", "auth_token": "y"}
    )
    assert isinstance(inst, TwilioAdapter)


def test_vector_store_factory_returns_faiss(tmp_faiss_index: str) -> None:
    inst = get_vector_store(
        {"provider": "faiss", "embedding_dim": 4, "index_path": tmp_faiss_index}
    )
    assert isinstance(inst, FAISSAdapter)


def test_unknown_provider_raises() -> None:
    with pytest.raises(UnknownProviderError):
        get_stt_provider({"provider": "made-up"})
