import os
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.analysis_storage_tool import AnalysisStorageTool

class QAAgent:
    def __init__(
        self,
        model: str = "gpt-4.1",
        persist_directory: str = "vector_store",
        doc_id: Optional[str] = None,
        k: int = 6,
        temperature: float = 0.3,
        max_history: int = 6,
    ):
        """
        QA Agent that answers user questions based on uploaded study material,
        PYQs, syllabus, or summarized YouTube videos.
        """
        self.model_name = model
        self.doc_id = doc_id
        self.k = k
        self.temperature = temperature
        self.max_history = max_history

        # LLM configuration
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # Connect to shared vector database (same as other agents)
        self.analysis_tool = AnalysisStorageTool(persist_directory)
        if not hasattr(self.analysis_tool, "vs") or self.analysis_tool.vs is None:
            raise RuntimeError("AnalysisStorageTool did not expose a vector store (vs).")

        search_kwargs: Dict[str, Any] = {"k": self.k}
        self.retriever = self.analysis_tool.vs.as_retriever(search_kwargs=search_kwargs)

        # Conversation memory
        self.history: List[Any] = []

        # Normalize document IDs for matching
        self._doc_id_candidates = self._normalize_doc_ids(doc_id)

    # ------------------------- PROMPT -------------------------

    @property
    def _system_prompt(self) -> str:
        return (
            "You are ExamPrepAI, a helpful and knowledgeable assistant for students.\n"
            "You answer questions using ONLY the provided CONTEXT, which may include notes, books, PYQs, "
            "syllabus, or YouTube lecture summaries.\n\n"
            "🎯 **GOALS:**\n"
            "- Explain topics clearly and accurately.\n"
            "- Give structured, step-by-step reasoning where needed.\n"
            "- Always cite which part of the context (book, notes, or video) you used.\n"
            "- If multiple related points exist, summarize them coherently.\n"
            "- If the question is not answerable from the context, say so honestly and suggest what the user can do next.\n\n"
            "🧠 **RULES:**\n"
            "1. NEVER invent facts or go beyond the given CONTEXT.\n"
            "2. Keep explanations conceptually rich but easy to understand for students.\n"
            "3. If question relates to PYQs or syllabus, highlight topic importance or exam trends.\n"
            "4. Use examples or definitions if they are available in context.\n"
            "5. Keep tone educational, supportive, and friendly."
        )

    # ------------------------- CONTEXT HANDLING -------------------------

    def _format_context(self, docs: List[Any]) -> str:
        """Combine retrieved docs into readable context block"""
        if not docs:
            return "(no relevant context found)"
        return "\n\n".join(
            f"[C{i+1}] Source: {d.metadata.get('source') or d.metadata.get('file_name') or 'unknown'}"
            + (f" | Page: {d.metadata.get('page')}" if d.metadata.get('page') else "")
            + f"\n{d.page_content or ''}"
            for i, d in enumerate(docs)
        )

    def _messages(self, question: str, context: str) -> List[Any]:
        msgs: List[Any] = [SystemMessage(content=self._system_prompt)]

        # Keep short memory context
        if self.history:
            msgs.extend(self.history[-self.max_history:])

        user_block = (
            f"CONTEXT (for answering):\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer using ONLY the CONTEXT above. Explain concepts clearly, include reasoning, "
            "and cite which context parts support your answer."
        )
        msgs.append(HumanMessage(content=user_block))
        return msgs

    # ------------------------- DOC MATCHING -------------------------

    def _normalize_doc_ids(self, raw_id: Optional[str]) -> set:
        if not raw_id:
            return set()
        bn = os.path.basename(raw_id)
        return {raw_id, bn, os.path.splitext(bn)[0]}

    def _doc_matches(self, metadata: dict) -> bool:
        """Match chunks belonging to this doc_id"""
        if not self.doc_id or not self._doc_id_candidates:
            return True
        for key in ("doc_id", "source", "file_name", "source_id", "source_filename"):
            val = metadata.get(key)
            if not val:
                continue
            vals = val if isinstance(val, (list, tuple)) else [val]
            if any(cand in str(v) for cand in self._doc_id_candidates for v in vals):
                return True
        return False

    # ------------------------- ANSWERING -------------------------

    def answer(self, question: str) -> str:
        """Main answering pipeline"""
        try:
            docs = self.retriever.get_relevant_documents(question)
        except Exception:
            # If retrieval fails, fallback to LLM without context
            warning = "⚠ Retrieval failed; answering without grounding.\n"
            msgs = [
                SystemMessage(content=self._system_prompt),
                *self.history[-self.max_history:],
                HumanMessage(content=f"{warning}QUESTION: {question}\nIf uncertain, say you don't know."),
            ]
            ai = self.llm.invoke(msgs)
            self.history.extend([HumanMessage(content=question), AIMessage(content=ai.content)])
            return ai.content

        # Filter by document ID
        warning_text = ""
        if self.doc_id:
            filtered = [d for d in docs if self._doc_matches(d.metadata or {})]
            if filtered:
                docs = filtered
            else:
                warning_text = (
                    "⚠ No chunks matched the requested document ID. Using most relevant ones.\n\n"
                )

        # Build and send messages
        context = self._format_context(docs)
        if warning_text:
            context = warning_text + context

        msgs = self._messages(question, context)
        ai = self.llm.invoke(msgs)

        # Maintain conversation history
        self.history.extend([HumanMessage(content=question), AIMessage(content=ai.content)])
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        return ai.content
