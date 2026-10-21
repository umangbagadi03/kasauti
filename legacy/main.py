from langgraph.graph import StateGraph, END
from state import ACHISState

from agents.ocr_agent import ocr_agent
from agents.ingredient_agent import ingredient_agent
from agents.risk_agent import risk_agent
from agents.knowledge_agent import knowledge_agent
from agents.reasoning_agent import reasoning_agent
from agents.guardrail_agent import guardrail_agent
from agents.decision_agent import decision_agent
from agents.explanation_agent import explanation_agent


def should_retry(state: ACHISState):
    retries = state.get("retry_count", 0)

    if not state.get("guardrail_passed") and retries < 2:
        state["retry_count"] = retries + 1
        return "reasoning"

    return "decision"


def need_knowledge(state: ACHISState):
    return "knowledge" if state.get("unknown_ingredients") else "reasoning"


graph = StateGraph(ACHISState)

graph.add_node("ocr", ocr_agent)
graph.add_node("ingredient", ingredient_agent)
graph.add_node("risk", risk_agent)
graph.add_node("knowledge", knowledge_agent)
graph.add_node("reasoning", reasoning_agent)                            
graph.add_node("guardrail", guardrail_agent)
graph.add_node("decision", decision_agent)
graph.add_node("explanation", explanation_agent)

graph.set_entry_point("ocr")

graph.add_edge("ocr", "ingredient")
graph.add_edge("ingredient", "risk")

# 🔥 Conditional knowledge usage
graph.add_conditional_edges("risk", need_knowledge, {
    "knowledge": "knowledge",
    "reasoning": "reasoning",
})

graph.add_edge("knowledge", "reasoning")

graph.add_edge("reasoning", "guardrail")

# 🔥 Safe retry loop
graph.add_conditional_edges("guardrail", should_retry, {
    "reasoning": "reasoning",
    "decision": "decision",
})

graph.add_edge("decision", "explanation")
graph.add_edge("explanation", END)

app = graph.compile()