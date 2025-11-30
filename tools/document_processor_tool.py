import os
import logging
from PIL import Image
import pytesseract
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.tools import tool   

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentProcessorTools:
    def __init__(
        self,
        persist_directory: str = "vector_store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 100
    ):
        """
        Handles text extraction, chunking, embedding, and storage into ChromaDB.
        """
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?"]
        )
        self.batch_size = batch_size


    def load_document(self, file_path: str):
        """Loads documents of type PDF, DOCX, or Image."""
        ext = file_path.split('.')[-1].lower()
        logger.info(f"Loading file: {file_path}")

        if ext == "pdf":
            loader = PyPDFLoader(file_path)
            return loader.load_and_split()

        elif ext == "docx":
            loader = Docx2txtLoader(file_path)
            return loader.load()

        elif ext in ["png", "jpg", "jpeg"]:
            text = pytesseract.image_to_string(Image.open(file_path), config="--psm 3")
            return [Document(page_content=text)]

        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def extract_text(self, file_path: str) -> str:
        """Extracts raw text from any supported document type."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        docs = self.load_document(file_path)
        if not docs:
            raise ValueError("No text extracted from document.")

        full_text = " ".join(doc.page_content for doc in docs)
        logger.info(f"Extracted text length: {len(full_text)} characters")
        return full_text

    def chunk_documents(self, documents):
        """Splits long documents into smaller overlapping chunks."""
        if len(documents) <= 1:
            return self.splitter.split_documents(documents)

        with ThreadPoolExecutor() as executor:
            chunk_lists = list(executor.map(
                lambda doc: self.splitter.split_documents([doc]), documents
            ))

        chunks = [chunk for sublist in chunk_lists for chunk in sublist]
        logger.info(f"Created {len(chunks)} text chunks.")
        return chunks


    def get_vectordb(self):
        """Initialize or load existing Chroma vector database."""
        try:
            if os.path.exists(self.persist_directory):
                vectordb = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                vectordb = Chroma.from_documents(
                    documents=[],
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            return vectordb
        except Exception as e:
            logger.warning(f"Vector DB initialization failed: {e}. Creating new DB.")
            return Chroma.from_documents(
                documents=[],
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

    def store_in_vectordb(self, chunks):
        """Stores chunked documents in Chroma vector store."""
        vectordb = self.get_vectordb()
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            vectordb.add_documents(batch)
        vectordb.persist()
        logger.info(f"Stored {len(chunks)} chunks in vector database.")
        return vectordb


    def process_document(self, file_path: str) -> dict:
        """Loads, extracts, chunks, and embeds a document into ChromaDB."""
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        try:
            logger.info(f"Processing document: {file_path}")
            docs = self.load_document(file_path)
            if not docs:
                return {"error": "No text extracted from document."}

            chunks = self.chunk_documents(docs)
            self.store_in_vectordb(chunks)
            extracted_text = " ".join(doc.page_content for doc in docs)

            return {
                "status": "success",
                "file_name": os.path.basename(file_path),
                "num_chunks": len(chunks),
                "vector_db_path": self.persist_directory,
                "extracted_text": extracted_text
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"error": str(e)}

    def store_metadata(self, content: str, meta_type: str, source_file: str):
        """Stores AI-generated content (summary, analysis, etc.) as metadata."""
        if not content.strip():
            return {"error": "Empty content, nothing to store."}

        doc = Document(
            page_content=content,
            metadata={"type": meta_type, "source": source_file}
        )
        self.store_in_vectordb([doc])

        return {
            "status": "success",
            "stored_type": meta_type,
            "source_file": source_file,
            "vector_db_path": self.persist_directory
        }

    def get_tools(self):
        """
        LangGraph requires standalone functions (not bound methods),
        so we wrap each tool into new decorated functions.
        """

        tools = []

        @tool("process_document")
        def _process_document(file_path: str):
            """Process a document: extract text, split chunks, and store embeddings in ChromaDB."""
            return self.process_document(file_path)
        tools.append(_process_document)

        @tool("extract_text")
        def _extract_text(file_path: str):
            """Extract raw text from a supported file without storing it."""
            return self.extract_text(file_path)
        tools.append(_extract_text)

        @tool("store_metadata")
        def _store_metadata(content: str, type: str = "generic", source: str = "unknown"):
            """Store AI-generated metadata (summaries, insights) inside the vector DB."""
            return self.store_metadata(content, type, source)
        tools.append(_store_metadata)

        return tools
