import os
from typing import List, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Vector DB sources
from tools.analysis_storage_tool import AnalysisStorageTool
from tools.document_processor_tool import DocumentProcessorTools


class QAAgent:
    """
    Answers user questions using BOTH:
    - Raw extracted text chunks (raw vector DB)
    - Stored analyses (summaries, PYQ trends, syllabus analysis, video summary)
      using the RETRIEVAL TOOL, not raw vector access
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        persist_directory: str = "vector_store",
        doc_id: Optional[str] = None,
        k: int = 6,
        temperature: float = 0.3,
        max_history: int = 6,
    ):
        self.model_name = model
        self.doc_id = doc_id
        self.k = k
        self.temperature = temperature
        self.max_history = max_history

        self.llm = ChatOpenAI(model=model, temperature=temperature)

        self.dp = DocumentProcessorTools(persist_directory)
        self.raw_vector_db = self.dp.get_vectordb()
        self.raw_retriever = self.raw_vector_db.as_retriever(search_kwargs={"k": k})

        self.analysis_tool = AnalysisStorageTool(persist_directory)
        self.retrieve_tool = self.analysis_tool.get_retrieval_tools()[0]

        self.history: List[Any] = []

        self._doc_id_candidates = self._normalize_doc_ids(doc_id)

    @property
    def _system_prompt(self) -> str:
        return (
            "You are ExamPrepAI, a knowledgeable assistant for students.\n"
            "You answer using ONLY the provided 'RAW TEXT CONTEXT' and 'ANALYSIS CONTEXT'.\n\n"
            "🎯 GOALS:\n"
            "- Explain clearly and accurately.\n"
            "- Highlight exam trends & important topics.\n"
            "- Prefer ANALYSIS context if present.\n"
            "- If context is weak, admit uncertainty.\n\n"
            "🧠 RULES:\n"
            "1. Never invent facts.\n"
            "2. If context is missing — say so.\n"
            "3. Use examples from context only.\n"
            "4. Keep tone: academic, helpful, exam-focused."
        )

    def _format_context(self, docs: List[Any], label: str) -> str:
        if not docs:
            return f"-- {label}: No context found --\n"

        out = [f"==== {label} ({len(docs)} docs) ===="]
        for i, d in enumerate(docs):
            src = d.metadata.get("source") or d.metadata.get("file_name") or "unknown"
            out.append(
                f"[{label}-{i+1}] Source: {src}\n{d.page_content}\n"
            )
        return "\n".join(out)

    def _messages(self, question: str, context: str) -> List[Any]:
        msgs: List[Any] = [SystemMessage(content=self._system_prompt)]
        msgs.extend(self.history[-self.max_history:])  # only last few messages
        msgs.append(HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"))
        return msgs

    def _normalize_doc_ids(self, raw_id: Optional[str]) -> set:
        if not raw_id:
            return set()
        bn = os.path.basename(raw_id)
        return {raw_id, bn, os.path.splitext(bn)[0]}

    def _doc_matches(self, metadata: dict) -> bool:
        """Filter by doc_id."""
        if not self.doc_id or not self._doc_id_candidates:
            return True
        for key in ("doc_id", "source", "file_name", "source_filename"):
            val = metadata.get(key)
            if val and any(c in str(val) for c in self._doc_id_candidates):
                return True
        return False

    def answer(self, question: str) -> str:

        try:
            raw_docs = self.raw_retriever.get_relevant_documents(question)
        except Exception:
            raw_docs = []

        if self.doc_id:
            raw_docs = [d for d in raw_docs if self._doc_matches(d.metadata)]

        try:
            tool_result = self.retrieve_tool.run(
                {
                    "query": question,
                    "k": self.k,
                    "filter": {"doc_id": self.doc_id} if self.doc_id else None
                }
            )

            analysis_docs = []
            for r in tool_result.get("results", []):
                analysis_docs.append(
                    type("Doc", (), {
                        "page_content": r["content"],
                        "metadata": r["metadata"]
                    })
                )
        except Exception:
            analysis_docs = []

        combined_docs = raw_docs + analysis_docs

        if not combined_docs:
            ai = self.llm.invoke([
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=f"No relevant context found.\nQUESTION: {question}")
            ])
            return ai.content

        context = (
            self._format_context(raw_docs, "RAW TEXT") +
            "\n\n" +
            self._format_context(analysis_docs, "STORED ANALYSIS")
        )

        msgs = self._messages(question, context)
        ai = self.llm.invoke(msgs)

        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=ai.content))

        return ai.content
