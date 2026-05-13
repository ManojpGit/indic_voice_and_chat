"""End-to-end task-completion benchmark for VoiceBotAgent.

Each ``TaskScenario`` declares scripted user turns (text-only — we bypass
the audio pipeline so the benchmark stays deterministic) plus the expected
final disposition and required slot values. The runner:

1. Builds a fresh ``VoiceBotAgent`` per scenario (via the supplied factory).
2. Feeds each scripted turn through a thin shim that writes synthetic STT
   results and lets the agent's LLM + dialogue layers run normally.
3. Scores: disposition match + slot fill rate + slot-value accuracy.

This is "task success" without the audio noise. Audio + endpointing is
covered by the latency benchmark; here we isolate dialogue logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Awaitable, Callable, Optional

from src.agents.state_machine import State
from src.agents.voicebot import VoiceBotAgent, TurnOutcome
from src.benchmarks.datasets import TaskScenario
from src.benchmarks.metrics import LatencyStats, latency_stats


@dataclass
class TaskScenarioResult:
    scenario_id: str
    completed: bool
    disposition_match: bool
    expected_disposition: str
    actual_action: str
    required_slot_fill_rate: float
    slot_value_match_rate: float
    turn_count: int
    duration_ms: float
    final_state: str


@dataclass
class TaskRunResult:
    scenario_count: int
    completion_rate: float
    disposition_match_rate: float
    avg_slot_fill_rate: float
    avg_slot_value_match_rate: float
    latency: LatencyStats
    per_scenario: list[TaskScenarioResult] = field(default_factory=list)


# Factory that builds a fresh agent per scenario.
AgentFactory = Callable[[TaskScenario], Awaitable[VoiceBotAgent]]

# Shim that runs one scripted user turn through the agent. Allows callers to
# inject a fake STT layer or to bypass it entirely.
TurnDriver = Callable[[VoiceBotAgent, str], Awaitable[TurnOutcome]]


async def run_task_benchmark(
    scenarios: list[TaskScenario],
    *,
    agent_factory: AgentFactory,
    turn_driver: TurnDriver,
) -> TaskRunResult:
    rows: list[TaskScenarioResult] = []
    latencies: list[float] = []
    completed_count = 0
    disposition_matches = 0
    slot_fill_rates: list[float] = []
    slot_match_rates: list[float] = []

    for scenario in scenarios:
        agent = await agent_factory(scenario)
        await agent.start()

        t0 = time.perf_counter()
        actual_action = ""
        for turn in scenario.user_turns:
            if agent.state.is_terminal:
                break
            if turn.role != "user" or not turn.content:
                continue
            outcome = await turn_driver(agent, turn.content)
            actual_action = outcome.response.action or actual_action
        dt = (time.perf_counter() - t0) * 1000.0

        # Disposition match: agent's final ``action`` (close_positive /
        # close_negative / transfer / schedule_callback / etc.) should map
        # to the scenario's expected disposition. Production wiring maps
        # actions → dispositions in the orchestrator; here we just compare
        # the action string against the expected disposition string.
        disposition_match = bool(
            scenario.expected_disposition
            and (scenario.expected_disposition == actual_action
                 or scenario.expected_disposition in (actual_action,))
        )
        if disposition_match:
            disposition_matches += 1

        required = scenario.required_slots or {}
        filled = [name for name in required if agent.slots.get(name) is not None]
        fill_rate = len(filled) / len(required) if required else 1.0
        slot_fill_rates.append(fill_rate)

        if required:
            matches = sum(
                1 for name, expected_value in required.items()
                if agent.slots.get(name) == expected_value
            )
            value_rate = matches / len(required)
        else:
            value_rate = 1.0
        slot_match_rates.append(value_rate)

        # "Completed" = both fill rate full and disposition matches.
        completed = disposition_match and fill_rate >= 1.0
        if completed:
            completed_count += 1
        latencies.append(dt)

        rows.append(TaskScenarioResult(
            scenario_id=scenario.id,
            completed=completed,
            disposition_match=disposition_match,
            expected_disposition=scenario.expected_disposition,
            actual_action=actual_action,
            required_slot_fill_rate=fill_rate,
            slot_value_match_rate=value_rate,
            turn_count=sum(1 for t in scenario.user_turns if t.role == "user"),
            duration_ms=dt,
            final_state=agent.state.state.value,
        ))

    n = max(len(scenarios), 1)
    return TaskRunResult(
        scenario_count=len(scenarios),
        completion_rate=completed_count / n,
        disposition_match_rate=disposition_matches / n,
        avg_slot_fill_rate=float(mean(slot_fill_rates)) if slot_fill_rates else 0.0,
        avg_slot_value_match_rate=float(mean(slot_match_rates)) if slot_match_rates else 0.0,
        latency=latency_stats(latencies),
        per_scenario=rows,
    )
