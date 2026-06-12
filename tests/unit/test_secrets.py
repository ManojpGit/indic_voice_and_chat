from __future__ import annotations

import pytest

from src.auth import secrets


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv(secrets.VOX_SECRET_KEY_ENV, secrets.generate_key())
    secrets.reset_cache_for_tests()
    yield
    secrets.reset_cache_for_tests()


def test_encrypt_decrypt_round_trip():
    token = secrets.encrypt("super-secret-twilio-token")
    assert token != "super-secret-twilio-token"          # actually encrypted
    assert secrets.decrypt(token) == "super-secret-twilio-token"


def test_encrypt_is_non_deterministic():
    # Fernet embeds a random IV → same plaintext encrypts differently each time.
    assert secrets.encrypt("x") != secrets.encrypt("x")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv(secrets.VOX_SECRET_KEY_ENV, raising=False)
    secrets.reset_cache_for_tests()
    with pytest.raises(secrets.SecretsError, match="VOX_SECRET_KEY"):
        secrets.encrypt("x")


def test_decrypt_with_wrong_key_raises(monkeypatch):
    token = secrets.encrypt("x")
    monkeypatch.setenv(secrets.VOX_SECRET_KEY_ENV, secrets.generate_key())  # rotate
    secrets.reset_cache_for_tests()
    with pytest.raises(secrets.SecretsError, match="could not decrypt"):
        secrets.decrypt(token)
