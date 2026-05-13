"""TenantContext — the per-request handle through the rest of the system.

A ``TenantContext`` carries the validated settings + a small helper for
resolving secrets. Every state-holding component that wants to behave
tenant-aware accepts one of these.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from src.config_tenant import TenantSettings


@dataclass(frozen=True)
class TenantContext:
    """Immutable per-request tenant handle."""

    settings: TenantSettings

    @property
    def id(self) -> str:
        return self.settings.id

    @property
    def slug(self) -> str:
        return self.settings.slug

    @property
    def name(self) -> str:
        return self.settings.name

    def secret(self, env_var: Optional[str]) -> Optional[str]:
        return self.settings.secret(env_var)


def hash_api_token(plaintext: str) -> str:
    """SHA-256 of the bearer token — the only form stored in the DB."""
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()
