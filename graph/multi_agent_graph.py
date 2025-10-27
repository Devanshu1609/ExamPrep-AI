# graph/multi_agent_graph.py
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage
import json
import re


def _to_snake(name: str) -> str:
    """Convert CamelCase or mixed → snake_case"""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.replace("__", "_").lower()


class MultiAgentGraph:
    def __init__(self, agents: dict):
        """
        agents: dict where keys are agent names like:
        {
            "supervisor_agent": supervisor_instance,
            "document_ingestion_agent": ingestion_instance,
            "summarizer_agent": summarizer_instance,
            "pyq_syllabus_analysis_agent": pyq_analysis_instance,
            "youtube_video_summarizer_agent": yt_summarizer_instance
        }
        """
        self.agents = agents
        self.graph = StateGraph(MessagesState)

    def build_graph(self):
        """Add all agents as nodes and define routing logic"""
        # Add nodes for all available agents
        for name, agent in self.agents.items():
            self.graph.add_node(name, agent)

        # After each specialized agent → return to supervisor for next step
        for name in self.agents.keys():
            if name != "supervisor_agent":
                self.graph.add_edge(name, "supervisor_agent")

        # Start → Supervisor
        self.graph.add_edge(START, "supervisor_agent")

        # Supervisor → Conditional routing based on decision
        self.graph.add_conditional_edges(
            "supervisor_agent",
            self.decide_next,
            {
                "document_ingestion_agent": "document_ingestion_agent",
                "summarizer_agent": "summarizer_agent",
                "pyq_syllabus_analysis_agent": "pyq_syllabus_analysis_agent",
                "youtube_video_summarizer_agent": "youtube_video_summarizer_agent",
                "end": END,
            }
        )

    def decide_next(self, state: MessagesState) -> str:
        """
        Reads Supervisor's last AI message (must be strict JSON):
        {
          "next_agent": "<agent_name or END>",
          "reason": "<reason>"
        }
        """
        last_msg = state["messages"][-1]

        if isinstance(last_msg, AIMessage):
            content = (last_msg.content or "").strip()
            try:
                decision = json.loads(content)
                next_agent = decision.get("next_agent", "").strip()
                reason = decision.get("reason", "No reason provided.")
                print(f"[Supervisor Decision] → {next_agent} | Reason: {reason}")
                return next_agent.lower() if next_agent else "end"
            except json.JSONDecodeError:
                print("[Error] Supervisor response not valid JSON. Ending workflow.")
                return "end"

        print("[Supervisor → END] No valid decision found.")
        return "end"

    def compile(self):
        """Compile the final state graph for execution"""
        return self.graph.compile()
