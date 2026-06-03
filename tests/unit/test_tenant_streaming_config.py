from __future__ import annotations

from src.config_tenant import TenantPipelineConfig, TenantStreamingSTTConfig


def test_streaming_stt_config_defaults_none():
    p = TenantPipelineConfig()
    assert p.stt_streaming is None


def test_streaming_stt_config_parses():
    p = TenantPipelineConfig(
        stt_streaming={
            "provider": "deepgram",
            "model": "nova-2",
            "language": "hi",
            "endpointing": 300,
            "utterance_end_ms": 1000,
            "api_key_env": "TENANT_DEV_DEEPGRAM_KEY",
        }
    )
    assert isinstance(p.stt_streaming, TenantStreamingSTTConfig)
    assert p.stt_streaming.provider == "deepgram"
    assert p.stt_streaming.endpointing == 300
    assert p.stt_streaming.api_key_env == "TENANT_DEV_DEEPGRAM_KEY"
