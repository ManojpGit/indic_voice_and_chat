# Multi-Tenant Refactor Plan

## Context

The framework is currently single-tenant. The provider-factory pattern (sarvam/groq/twilio swappable via one YAML) works at the *provider* level, but there's exactly one `PipelineConfig`, one Redis namespace, one FAISS index, one DND list, one webhook registry, one orchestrator. Phase 5's per-campaign `pipeline_override` exists in the YAML schema but isn't actually consumed.

No production data exists yet, so we can rewrite the alembic baseline migration cleanly — no backfill, no compatibility shims, no data-migration code.

**End state**: every entity that holds tenant-specific state carries a `tenant_id`. Provider clients are built per-tenant from a `TenantSettings` layer that overrides the global YAML defaults. API routes resolve a `TenantContext` from auth and thread it through orchestrator → agent → pipeline. Tests parameterize by tenant.

## Scope

**In**:
- `Tenant` + `TenantPhoneNumber` data model
- Per-tenant scoping of all state-holding components (Redis, FAISS, DND, rate limits, webhooks, sessions)
- Auth middleware → `TenantContext` (bearer-token, tenant lookup by Twilio `To` number)
- Tenant-aware provider factories (clients cached per tenant)
- Per-tenant orchestrator / retriever / chatbot factory wiring
- Lifespan bootstrap that loads every configured tenant

**Out (deferred)**:
- Encrypted secret vault (secrets stay in env vars referenced by name in tenant YAML)
- Per-tenant quotas / billing / rate limiting beyond what the call scheduler already enforces
- Tenant admin UI / self-serve onboarding
- RBAC beyond simple bearer-token auth
- Tenant suspension/deletion workflows
- Multi-region / data-residency routing

## Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Tenant config storage | YAML files in `config/tenants/<slug>.yaml` + minimal `tenants` table for FK integrity | Version-controlled config, no plaintext-in-DB secrets, simple bootstrap |
| API key storage | Tenant YAML references env var names (`api_key_env: TENANT_ACME_SARVAM_KEY`), never raw values | Secrets stay out of repo and DB until a real vault lands |
| Redis topology | Single instance, key prefix `tenant:{tid}:` | Simpler ops than per-tenant DBs; FAISS pattern matches |
| Postgres topology | Single DB, `tenant_id` FK on scoped tables | Standard SaaS pattern; cheap joins |
| FAISS topology | Single base dir, per-tenant subdirectory `data/faiss/<tid>/` | One file per tenant keeps cleanup trivial |
| Inbound call routing | Twilio `To` number → tenant lookup in `tenant_phone_numbers` table | Twilio webhook already supplies `To`; no DNS games |
| Migration | Rewrite `0001_initial.py` to include tenants from the start | No data exists; cleanest possible baseline |

## Milestones

Each milestone has its own verification gate. Don't move on until the gate is green.

### MT.1 — Tenant domain model

- `src/models/tenant.py`: `Tenant`, `TenantPhoneNumber`, `TenantApiKey` ORM
- Rewrite `alembic/versions/0001_initial.py`:
  - New `tenants`, `tenant_phone_numbers`, `tenant_api_keys` tables
  - `tenant_id VARCHAR(50) NOT NULL` FK on `campaigns`, `leads`, `conversations`, `kb_documents`
  - `tenant_id VARCHAR(50)` (nullable — system-wide runs allowed) on `benchmark_runs`
- `src/campaign/models.py`: add `tenant_id` to `Campaign`, `Lead`, `CallResult`
- **Gate**: `alembic upgrade head` clean on fresh DB; existing `test_models.py` adapted to seed a tenant; all green.

### MT.2 — Tenant settings + config loader

- `src/config.py`: new `TenantSettings` (provider overrides per layer, compliance overrides, branding, webhook secret, default language)
- `config/tenants/` directory; one YAML file per tenant
- `src/config.py::load_tenant(slug)` reads YAML, resolves env-var references for API keys, returns a fully-validated `TenantSettings`
- Provider config slice: tenant settings override global defaults field-by-field (so a tenant can override just `tts.voice_id` without redeclaring the whole stack)
- Sample: `config/tenants/example.yaml`
- **Gate**: `tests/unit/test_tenant_config.py` covers happy path, missing-env-var, unknown-slug, partial override.

### MT.3 — TenantContext + auth middleware

- `src/auth/context.py`: `TenantContext` dataclass — `id`, `slug`, `settings`, `resolved_secrets`, helpers `pipeline_config_for(layer)`
- `src/auth/middleware.py`: FastAPI dependency `current_tenant(request)` resolves from:
  - `Authorization: Bearer <token>` → SHA-256 hash → `tenant_api_keys.token_hash` row
  - For Twilio voice webhook: lookup by `To` form param in `tenant_phone_numbers`
  - For Twilio Media Streams WS: tenant carried in query param (set by the voice webhook's TwiML response)
- Returns 401 on missing/invalid auth; 403 on cross-tenant resource access
- **Gate**: unit tests for each auth path; 401 / 403 paths exercised.

### MT.4 — Tenant-aware provider factories

- `TenantProviders` registry: lazily builds + caches per-tenant provider clients (`stt`, `llm`, `tts`, `telephony`, `vector_store`) keyed by `(tenant_id, layer)`
- `src/providers/__init__.py`: `get_stt_provider(tenant_ctx)` etc. — pulls config slice from tenant, API key from `resolved_secrets`
- Eviction policy: drop cached clients when tenant config reloads (not in this phase, but the interface allows it)
- **Gate**: new test — two tenants resolve to distinct provider instances with distinct API keys; calling `get_stt_provider` twice for the same tenant returns the same instance.

### MT.5 — Scope state-holding components

For each component, the key change is "add `tenant_id` parameter; namespace internal state":

| Component | Change |
|---|---|
| `SessionStore` | Constructor takes `tenant_id`; Redis keys become `tenant:{tid}:session:{sid}:state\|history\|slots` |
| `FAISSAdapter` | `index_path` resolves to `{base}/{tenant_id}/` |
| `HybridRetriever` | One instance per tenant; `RetrieverRegistry` resolves by tenant_id |
| `InMemoryDNDStore` / `DNDFilter` | Per-tenant; held in `DNDRegistry` |
| `CallScheduler` | `_dispatched_at` becomes `dict[tenant_id, deque]`; rate limit applied per tenant |
| `WebhookManager` | Per-tenant registry; cross-tenant events never fan out |
| `WhatsAppHandoff` | Picks chat channel from tenant config; template registry merges tenant overrides over defaults |
| `EventBus` | Events carry `tenant_id` in payload; per-tenant subscribers filter on it |

- **Gate**: cross-tenant isolation tests — operations on tenant A leave tenant B's state untouched; new fixture `two_tenants` parameterizes tests.

### MT.6 — Tenant-aware orchestrator + agents

- `CampaignOrchestrator`: resolves tenant from `campaign.tenant_id`, builds providers via `TenantProviders`, uses tenant's DND store + scheduler bucket
- `VoiceBotAgent` + `ChatBotAgent` factories: accept `TenantContext`, use tenant's script overrides if specified (campaigns can still further override per-campaign)
- `FakeChatChannel` / `IChatChannel`: tenant-aware via event payload
- **Gate**: orchestrator E2E test — two campaigns under different tenants run concurrently; each uses its own provider clients, DND list, webhook deliveries.

### MT.7 — API routes resolve tenant everywhere

- Every route under `/api/v1/*` (except `/health` and `/api/v1/telephony/*` which has its own resolution) gets `tenant: TenantContext = Depends(current_tenant)`
- All campaign / lead / conversation / knowledge / chat queries filter by `tenant.id`
- `set_*_factory` patterns: the factory now takes `TenantContext` and returns the per-tenant instance from the registry
- Twilio voice webhook: tenant resolved from `To` form param
- Twilio Media Streams WS: tenant resolved from query param set by the voice TwiML
- **Gate**: route tests adapted to inject a tenant header / bearer token; cross-tenant access returns 403 (or 404, depending on whether existence is leaked).

### MT.8 — Lifespan bootstrap

- `src/main.py` lifespan:
  1. Read tenant manifest (list of slugs from `config/tenants/`)
  2. For each tenant: load `TenantSettings`, register in `TenantRegistry`
  3. Wire per-tenant `TenantProviders`, `RetrieverRegistry`, `DNDRegistry`, `WebhookManager`, orchestrator dispatch, chatbot factory, twilio bridge factory
  4. Tear down on shutdown (close engine, redis pool, dispose provider clients)
- `scripts/check_providers.py` extended to print provider routing per tenant
- `/health` reports `{tenants: [{slug, providers: {...}}, ...]}`
- **Gate**: app starts with two configured tenants; smoke test verifies `/health` shape; `check_providers.py` lists both tenants.

### MT.9 — End-to-end multi-tenant integration test

- Configure two tenants: `acme` and `globex` — both use sarvam/groq/sarvam stack but with distinct API keys + distinct DND lists + distinct webhook URLs
- Both upload leads, both run campaigns concurrently
- Verify cross-tenant isolation:
  - Provider clients are distinct instances (different API keys used per provider call)
  - Redis sessions don't collide
  - FAISS indexes are separate directories
  - Webhook deliveries fan out only to the originating tenant's registered URLs
  - DND blocks on acme's number don't affect globex
- **Gate**: integration test green; coverage ≥ 90%; all 397 existing tests still green.

## Critical files to create or change

```
NEW
  src/models/tenant.py
  src/auth/__init__.py
  src/auth/context.py
  src/auth/middleware.py
  src/auth/registry.py                  # TenantRegistry, TenantProviders, RetrieverRegistry, DNDRegistry
  config/tenants/example.yaml
  tests/unit/test_tenant_config.py
  tests/unit/test_tenant_auth.py
  tests/unit/test_tenant_registry.py
  tests/integration/test_multi_tenant_e2e.py

REWRITE
  alembic/versions/0001_initial.py      # rewritten with tenant_id everywhere
  src/config.py                          # add TenantSettings + loader
  src/providers/__init__.py             # accept TenantContext
  src/dialogue/context.py               # tenant-namespaced Redis keys
  src/providers/vector_store/faiss_store.py   # per-tenant index path
  src/rag/retriever.py                  # per-tenant via registry (or constructor)
  src/campaign/models.py                # tenant_id everywhere
  src/campaign/dnd_filter.py            # per-tenant DND store
  src/campaign/scheduler.py             # per-tenant rate-limit window
  src/campaign/orchestrator.py          # resolve tenant from campaign
  src/integration/webhooks.py           # per-tenant registry
  src/integration/event_bus.py          # tenant_id on event payload
  src/integration/handoff.py            # tenant-aware channel + templates
  src/api/__init__.py                   # all routes accept tenant dep
  src/api/campaigns.py                  # filter by tenant
  src/api/chat.py                       # filter by tenant
  src/api/knowledge.py                  # filter by tenant
  src/api/webhooks_routes.py            # per-tenant webhooks
  src/api/telephony_hooks.py            # tenant resolution from To number / WS query
  src/api/benchmarks.py                 # optional tenant scoping
  src/main.py                           # lifespan loads tenants
  scripts/check_providers.py            # per-tenant wiring check

ADAPT
  tests/conftest.py                     # tenant fixtures
  every existing route + adapter test   # inject tenant
```

## Effort estimate

Based on Phase 5 (orchestrator + state mgmt) which was similar scope:

- MT.1–MT.2 (models + config): 0.5 day
- MT.3 (auth): 0.5 day
- MT.4 (factories): 0.5 day
- MT.5 (state scoping): 1 day — touches the most files
- MT.6 (orchestrator + agents): 0.5 day
- MT.7 (API routes): 0.5 day — mostly mechanical
- MT.8 (bootstrap): 0.5 day
- MT.9 (E2E test): 0.5 day

**Total: ~4 days of focused work.** Test adaptation across the existing 397 tests is the long pole — most need a tenant fixture parameter added.

## End-to-end verification (after MT.9)

1. `alembic upgrade head` on fresh Postgres — clean
2. `pytest tests/ -v` — all green, including new multi-tenant E2E
3. `pytest --cov=src` — coverage ≥ 90%
4. `python scripts/check_providers.py` — lists each configured tenant + its provider routing
5. `uvicorn src.main:app --port 8000` + `curl localhost:8000/health` — returns tenant count + per-tenant provider summary
6. Manual: two configured tenants, hit `POST /api/v1/campaigns` with each tenant's bearer token — confirm responses are tenant-scoped, 403 on cross-tenant access

## Decisions (locked)

1. **WS tenant resolution**: TwiML embeds `?tenant={slug}` in the `<Stream url=...>`; the WS handler reads it on connect.
2. **LLM keys**: each tenant supplies its own (no platform-wide fallback). Tenant YAML must reference a real env var; missing env var on startup is fatal for that tenant.
3. **Benchmarks**: platform-admin only. No `tenant_id` on `benchmark_runs`. `/api/v1/benchmarks/*` requires admin auth, not tenant auth.
4. **WhatsApp inbound**: signature verification against the tenant's webhook secret. Resolve tenant from a route segment or `X-Tenant-Slug` header before verifying — incoming webhooks are unauthenticated until the signature checks out.
