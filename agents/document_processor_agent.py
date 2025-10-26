from langgraph.prebuilt import create_react_agent
from tools.document_processor_tool import DocumentProcessorTools

class DocumentProcessorAgent:
    def __init__(self, model="openai:gpt-4.1", persist_directory="vector_store"):
        self.model = model
        self.tools = DocumentProcessorTools(persist_directory=persist_directory).get_tools()
        self.name = "document_processor_agent"
        self.description = "Extracts and processes text from documents efficiently."

    def create_agent(self):
        return create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=(
                "You are DocumentProcessorAgent, an expert AI assistant for processing and storing documents into a vector database.\n\n"
                "### INSTRUCTIONS ###\n"
                "1. Accept PDF and DOCX file paths for ingestion.\n"
                "2. Use your tools to:\n"
                "   - Load and extract full text from the document efficiently.\n"
                "   - Chunk text for efficient vector DB storage.\n"
                "   - Store chunks in the vector DB using batch embeddings.\n"
                "3. After ingestion, return a structured JSON:\n"
                "   {\n"
                "     \"file_name\": <original file name>,\n"
                "     \"num_chunks\": <number of chunks stored>,\n"
                "     \"vector_db_path\": <path to vector DB>,\n"
                "     \"extracted_text\": <full raw extracted text>\n"
                "   }\n"
                "4. Do NOT summarize or alter the extracted text — it will be passed to another agent.\n"
                "5. Keep responses machine-readable for downstream agents."
            ),
            name="document_processor_agent"
        )
