# graph/multi_agent_graph.py
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage
import json

class MultiAgentGraph:
    def __init__(self, agents: dict):
        self.agents = agents
        self.graph = StateGraph(MessagesState)

    def build_graph(self):
        for name, agent in self.agents.items():
            self.graph.add_node(name, agent)

        for name in self.agents.keys():
            if name != "supervisor_agent":
                self.graph.add_edge(name, "supervisor_agent")

        self.graph.add_edge(START, "supervisor_agent")

        self.graph.add_conditional_edges(
            "supervisor_agent",
            self.decide_next,
            {
                "document_ingestion_agent": "document_ingestion_agent",
                "summarizer_agent": "summarizer_agent",
                "pyq_syllabus_analysis_agent": "pyq_syllabus_analysis_agent",
                "youtube_video_summarizer_agent": "youtube_video_summarizer_agent",
                "store_analysis_agent": "store_analysis_agent",
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
        return self.graph.compile()
