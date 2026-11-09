"""ARIA LangGraph bridge: planner selection and mandatory-gate orchestration.

This module deliberately receives the existing backend functions as dependencies.
It does not import the ARIA backend, initialise external clients, or execute tools.
The first migration slice preserves the existing planner/safety helper behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph


BackendFunctions = Mapping[str, Callable[..., Any]]


class PlannerSelectionState(TypedDict, total=False):
    context: dict[str, Any]
    llm_plan: Optional[dict[str, Any]]
    next_tool: Optional[str]
    selection_source: str
    planner_reason: str
    planner_finish_requested: bool
    planner_finish_honored: bool
    finish_gate_reason: str


def build_aria_planner_selection_graph(
    backend: BackendFunctions,
):
    """Build the first ARIA graph slice using the backend's existing helpers.

    Required backend keys:
      select_planner_tool
      planner_requests_finish
      planner_finish_is_safe
      adaptive_fallback_next_tool
      enforce_mandatory_investigation_gates

    The graph does not execute any banking tool. The existing run_agent() flow
    remains responsible for subsequent special-case overrides, registry checks,
    clarification, tool authorization, execution, audit and SSE events.
    """
    required = (
        "select_planner_tool",
        "planner_requests_finish",
        "planner_finish_is_safe",
        "adaptive_fallback_next_tool",
        "enforce_mandatory_investigation_gates",
    )
    missing = [name for name in required if not callable(backend.get(name))]
    if missing:
        raise ValueError(
            "Missing callable backend dependencies: " + ", ".join(missing)
        )

    def validate_proposal(state: PlannerSelectionState) -> dict[str, Any]:
        context = state["context"]
        plan = state.get("llm_plan")
        next_tool = backend["select_planner_tool"](context, plan)
        return {"next_tool": next_tool}

    def evaluate_finish_request(state: PlannerSelectionState) -> dict[str, Any]:
        context = state["context"]
        plan = state.get("llm_plan")
        finish_requested = bool(backend["planner_requests_finish"](plan))
        finish_honored = False
        finish_reason = ""
        planner_reason = str(state.get("planner_reason") or "")

        if finish_requested and not state.get("next_tool"):
            finish_honored, finish_reason = backend["planner_finish_is_safe"](
                context
            )
            finish_honored = bool(finish_honored)
            if finish_honored:
                planner_reason = planner_reason or str(finish_reason or "")

        return {
            "planner_finish_requested": finish_requested,
            "planner_finish_honored": finish_honored,
            "finish_gate_reason": str(finish_reason or ""),
            "planner_reason": planner_reason,
            "selection_source": (
                "llm_planner_finish"
                if finish_honored
                else str(state.get("selection_source") or "llm_planner")
            ),
        }

    def choose_route(state: PlannerSelectionState) -> str:
        if state.get("planner_finish_honored"):
            return "mandatory_gates"
        if not state.get("next_tool"):
            return "fallback"
        return "mandatory_gates"

    def select_fallback(state: PlannerSelectionState) -> dict[str, Any]:
        next_tool = backend["adaptive_fallback_next_tool"](state["context"])
        return {
            "next_tool": next_tool,
            "selection_source": "state_aware_fallback",
        }

    def apply_mandatory_gates(state: PlannerSelectionState) -> dict[str, Any]:
        before = state.get("next_tool")
        next_tool = backend["enforce_mandatory_investigation_gates"](
            state["context"],
            before,
            allow_fallback=not bool(state.get("planner_finish_honored")),
        )
        selection_source = str(state.get("selection_source") or "llm_planner")
        if next_tool != before:
            selection_source = "mandatory_evidence_or_policy_gate"
        return {
            "next_tool": next_tool,
            "selection_source": selection_source,
        }

    builder = StateGraph(PlannerSelectionState)
    builder.add_node("validate_proposal", validate_proposal)
    builder.add_node("evaluate_finish_request", evaluate_finish_request)
    builder.add_node("select_fallback", select_fallback)
    builder.add_node("apply_mandatory_gates", apply_mandatory_gates)

    builder.add_edge(START, "validate_proposal")
    builder.add_edge("validate_proposal", "evaluate_finish_request")
    builder.add_conditional_edges(
        "evaluate_finish_request",
        choose_route,
        {
            "fallback": "select_fallback",
            "mandatory_gates": "apply_mandatory_gates",
        },
    )
    builder.add_edge("select_fallback", "apply_mandatory_gates")
    builder.add_edge("apply_mandatory_gates", END)

    return builder.compile()
