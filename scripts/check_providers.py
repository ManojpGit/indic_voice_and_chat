"""Provider factory wiring smoke test.

Loads ``config/default.yaml`` and instantiates every configured adapter via
the factories. No external calls are made — only constructor execution.
Useful for catching wiring breakage without spinning up the full app.

Run:  python scripts/check_providers.py
"""

from __future__ import annotations

import os
import sys

# Provide dummy env vars so adapter constructors don't fail when checking
# wiring locally without real keys.
os.environ.setdefault("SARVAM_API_KEY", "dummy")
os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("TWILIO_ACCOUNT_SID", "ACdummy")
os.environ.setdefault("TWILIO_AUTH_TOKEN", "dummy")

from src.config import load_settings  # noqa: E402
from src.providers import (  # noqa: E402
    get_llm_provider,
    get_stt_provider,
    get_telephony_provider,
    get_tts_provider,
    get_vector_store,
)


def main() -> int:
    settings = load_settings()
    p = settings.pipeline

    instances = {
        "stt": get_stt_provider(p.stt.model_dump()),
        "llm": get_llm_provider(p.llm.model_dump()),
        "tts": get_tts_provider(p.tts.model_dump()),
        "telephony": get_telephony_provider(
            {
                **p.telephony.model_dump(),
                "account_sid": os.environ["TWILIO_ACCOUNT_SID"],
                "auth_token": os.environ["TWILIO_AUTH_TOKEN"],
            }
        ),
        "vector_store": get_vector_store(p.vector_store.model_dump()),
    }

    print("Provider factory wiring:")
    for layer, inst in instances.items():
        print(f"  {layer:14s} -> {inst.__class__.__module__}.{inst.__class__.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
