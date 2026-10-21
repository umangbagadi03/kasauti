"""The Kasauti pipeline as a LangGraph graph.

    barcode? ──► lookup ──┐ (found)
       │                  ▼
       └──────► extract ─► analyze ─► (unknown ingredients?) ─► enrich ─► analyze
                                  └─────────────────────────────► score ─► explain

LLM calls: extract (1, cached per product), enrich (≤1, cached per ingredient), explain (1).
Everything that decides the verdict (analyze, score) is deterministic.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from .nodes import llm_from
from .nodes.analyze import analyze_node
from .nodes.enrich import enrich_node
from .nodes.explain import explain_node
from .nodes.extract import extract_node
from .nodes.lookup import lookup_node
from .nodes.score import score_node
from .state import GraphState


def _start(state: GraphState) -> str:
    return "lookup" if state.get("barcode") else "extract"


def _after_lookup(state: GraphState) -> str:
    if state.get("label"):
        return "analyze"
    return "extract" if state.get("image") or state.get("text") else "score"


def _after_analyze(state: GraphState, config) -> str:
    if state.get("unknown") and not state.get("enriched") and llm_from(config):
        return "enrich"
    return "score"


@lru_cache(maxsize=1)
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("lookup", lookup_node)
    g.add_node("extract", extract_node)
    g.add_node("analyze", analyze_node)
    g.add_node("enrich", enrich_node)
    g.add_node("score", score_node)
    g.add_node("explain", explain_node)

    g.add_conditional_edges(START, _start, ["lookup", "extract"])
    g.add_conditional_edges("lookup", _after_lookup, ["analyze", "extract", "score"])
    g.add_edge("extract", "analyze")
    g.add_conditional_edges("analyze", _after_analyze, ["enrich", "score"])
    g.add_edge("enrich", "analyze")
    g.add_edge("score", "explain")
    g.add_edge("explain", END)
    return g.compile()
