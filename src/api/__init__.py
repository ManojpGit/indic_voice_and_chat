"""FastAPI routes — central router that aggregates all sub-routers."""

from fastapi import APIRouter

from src.api import (
    benchmarks,
    campaigns,
    chat,
    config_routes,
    conversations,
    knowledge,
    sessions,
    telephony_hooks,
    webhooks_routes,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(sessions.router)
api_router.include_router(campaigns.router)
api_router.include_router(config_routes.router)
api_router.include_router(conversations.router)
api_router.include_router(knowledge.router)
api_router.include_router(webhooks_routes.router)
api_router.include_router(benchmarks.router)
api_router.include_router(chat.router)
api_router.include_router(telephony_hooks.router)
