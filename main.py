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
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

load_dotenv()

app = FastAPI(title="Legal Document Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_state = {
    "file_path": None,
    "youtube_url": None,
    "chat_history": []
}

UPLOAD_FILE_PATH = r"uploads\CN_pyq.pdf"
VECTOR_DB_DIR = "vector_store"

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

agent_graph = MultiAgentGraph(agents)
agent_graph.build_graph()
compiled_graph = agent_graph.compile()

qa_agent = QAAgent(persist_directory=VECTOR_DB_DIR)

def run_graph(file_path: str):
    print(f"[INFO] Starting workflow via graph for: {file_path}")
    input_state = {
        "messages": [HumanMessage(content=f"New file uploaded: {file_path}")]
    }

    agent_response_fragments = []
    try:
        for chunk in compiled_graph.stream(input_state):
            pretty_print_messages(chunk, last_message=True)
            if "messages" in chunk and chunk["messages"]:
                for m in chunk["messages"]:
                    if m.content:
                        agent_response_fragments.append(m.content)
    except Exception as e:
        print(f"[ERROR] Graph execution failed: {e}")

    final_message = "\n".join(agent_response_fragments).strip()
    if final_message:
        print(f"\n[INFO] Workflow completed. Final supervisor output:\n{final_message}\n")


class ChatRequest(BaseModel):
    message: str

@app.post("/upload")
async def upload_file_or_url(
    file: UploadFile = File(None),
    youtube_url: str = Form(None)
):
    if file:
        file_location = f"uploads/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        session_state["file_path"] = file_location
        run_graph(file_location)
        return {"status": "success", "message": f"File '{file.filename}' uploaded and processed."}
    elif youtube_url:
        session_state["youtube_url"] = youtube_url
        try:
            yt_agent = YouTubeSummarizerAgent(persist_directory=VECTOR_DB_DIR)
            result = yt_agent.summarize(youtube_url)
            message = f"YouTube URL '{youtube_url}' processed. Summary: {result}"
        except Exception as e:
            message = f"YouTube URL '{youtube_url}' received, but processing failed: {e}"
        return {"status": "success", "message": message}
    else:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No file or YouTube URL provided."})

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    question = request.message.strip()
    if not question:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Empty message."})
    session_state["chat_history"].append(("user", question))
    messages = []
    for role, msg in session_state["chat_history"]:
        if role == "user":
            messages.append(HumanMessage(content=msg))
        elif role == "agent":
            messages.append(AIMessage(content=msg))
    answer = qa_agent.answer(question)
    session_state["chat_history"].append(("agent", answer))
    return {"status": "success", "answer": answer}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
