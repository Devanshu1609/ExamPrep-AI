# main.py
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from graph.multi_agent_graph import MultiAgentGraph
from agents.supervisor_agent import SupervisorAgent
from agents.document_processor_agent import DocumentProcessorAgent
from agents.summarizer_agent import SummarizerAgent
from agents.pyq_syllabus_analyser_agent import PYQSyllabusAnalyserAgent
from agents.youtube_summarizer_agent import YouTubeSummarizerAgent
from agents.StoreAnalysisAgent import StoreAnalysisAgent
from agents.qa_agent import QAAgent
from utils.message_utils import pretty_print_messages  # optional helper to print nicely
from fastapi import FastAPI
load_dotenv()

app = FastAPI(title="Legal Document Assistant")

# ------------------- CONFIG ------------------- #
UPLOAD_FILE_PATH = r"uploads\CN_pyq.pdf"  # <-- Change this to your test file
VECTOR_DB_DIR = "vector_store"

# ------------------- Initialize Agents ------------------- #
doc_ingestion_agent = DocumentProcessorAgent(persist_directory=VECTOR_DB_DIR).create_agent()
summarizer_agent = SummarizerAgent(persist_directory=VECTOR_DB_DIR).create_agent()
pyq_analysis_agent = PYQSyllabusAnalyserAgent(persist_directory=VECTOR_DB_DIR).create_agent()
youtube_agent = YouTubeSummarizerAgent(persist_directory=VECTOR_DB_DIR).create_agent()
supervisor_agent = SupervisorAgent().create_agent()
analysis_storage_agent = StoreAnalysisAgent(persist_directory=VECTOR_DB_DIR).create_agent()

agents = {
    "supervisor_agent": supervisor_agent,
    "document_ingestion_agent": doc_ingestion_agent,
    "summarizer_agent": summarizer_agent,
    "pyq_syllabus_analysis_agent": pyq_analysis_agent,
    "youtube_video_summarizer_agent": youtube_agent,
    "store_analysis_agent": analysis_storage_agent,
}

# ------------------- Build Multi-Agent Graph ------------------- #
agent_graph = MultiAgentGraph(agents)
agent_graph.build_graph()
compiled_graph = agent_graph.compile()

# ------------------- QA Agent for chat ------------------- #
qa_agent = QAAgent(persist_directory=VECTOR_DB_DIR)

# ------------------- Supervisor Graph Execution ------------------- #
def run_graph(file_path: str):
    print(f"[INFO] Starting workflow via graph for: {file_path}")
    # Initial state with uploaded file info
    input_state = {
        "messages": [HumanMessage(content=f"New file uploaded: {file_path}")]
    }

    # Stream agent responses via compiled graph
    agent_response_fragments = []
    try:
        for chunk in compiled_graph.stream(input_state):
            pretty_print_messages(chunk, last_message=True)
            # Collect any AI message content
            if "messages" in chunk and chunk["messages"]:
                for m in chunk["messages"]:
                    if m.content:
                        agent_response_fragments.append(m.content)
    except Exception as e:
        print(f"[ERROR] Graph execution failed: {e}")

    final_message = "\n".join(agent_response_fragments).strip()
    if final_message:
        print(f"\n[INFO] Workflow completed. Final supervisor output:\n{final_message}\n")


# ------------------- CLI Chat ------------------- #
def chat_loop():
    print("\n=== AI Exam Prep Chatbot ===")
    print("You can now ask questions based on the uploaded document/video.")
    print("Type 'exit' to quit.\n")

    chat_history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        chat_history.append(("user", question))

        # Prepare messages for QA agent context
        messages = []
        for role, msg in chat_history:
            if role == "user":
                messages.append(HumanMessage(content=msg))
            elif role == "agent":
                messages.append(AIMessage(content=msg))

        answer = qa_agent.answer(question)
        print(f"AI: {answer}\n")

        chat_history.append(("agent", answer))


# ------------------- MAIN ------------------- #
if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FILE_PATH):
        print(f"[ERROR] File not found: {UPLOAD_FILE_PATH}")
    else:
        # Run the workflow graph first
        run_graph(UPLOAD_FILE_PATH)

        # Start CLI chat interface
        chat_loop()
