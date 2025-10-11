# tools/analysis_storage_tool.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from pydantic import BaseModel, Field

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LCDocument
from langchain_core.tools import StructuredTool


class _StoreArgs(BaseModel):
    agent_name: str = Field(..., description="Name of the agent storing the result, e.g., 'summarizer_agent'.")
    result_type: str = Field(..., description="Type of result, e.g., 'summary', 'risk_analysis', 'clauses'.")
    result: Union[Dict[str, Any], List[Any], str] = Field(..., description="The analysis content to store (JSON or string).")
    doc_id: Optional[str] = Field(None, description="Optional document ID that this analysis refers to.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional extra metadata to index with the record.")


class _RetrieveArgs(BaseModel):
    query: str = Field(..., description="User query to retrieve relevant analyses.")
    k: int = Field(5, description="Number of top matches to return.")
    filter: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata filter, e.g., {'type': 'summary'} or {'doc_id': 'abc'}."
    )


class AnalysisStorageTool:
    """
    Provides structured tools for:
    - Storing agent outputs (summary/risk/clauses/etc.) in Chroma with metadata
    - Retrieving relevant analyses for Q&A
    """

    def __init__(self, persist_directory: str = "vector_store"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Create or load persistent Chroma store
        self.vs = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

        # Build tools
        self._store_tool = StructuredTool.from_function(
            name="store_analysis_result",
            description=(
                "Store an analysis result (summary, risk analysis, clauses, etc.) into the vector database "
                "with rich metadata for later retrieval."
            ),
            func=self._store_impl,
            args_schema=_StoreArgs,
            return_direct=False,
        )

        self._retrieve_tool = StructuredTool.from_function(
            name="retrieve_analysis",
            description=(
                "Retrieve relevant stored analyses (summaries, clauses, risk analysis) from the vector database. "
                "Optionally filter by metadata like {'type': 'summary'} or {'doc_id': 'XYZ'}."
            ),
            func=self._retrieve_impl,
            args_schema=_RetrieveArgs,
            return_direct=False,
        )

    # ---------- Tool implementations ----------
    def _store_impl(
        self,
        agent_name: str,
        result_type: str,
        result: Union[Dict[str, Any], List[Any], str],
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Normalize text representation (for vector embedding)
        if isinstance(result, (dict, list)):
            import json
            text_repr = json.dumps(result, ensure_ascii=False)
        else:
            text_repr = str(result)

        meta = {
            "agent_name": agent_name,
            "type": result_type,
            "doc_id": doc_id,
            "timestamp": timestamp,
        }
        if metadata:
            meta.update(metadata)

        # Add to vector store
        self.vs.add_documents([LCDocument(page_content=text_repr, metadata=meta)])
        self.vs.persist()

        return f"Stored analysis result: type='{result_type}', agent='{agent_name}', doc_id='{doc_id}'"

    def _retrieve_impl(self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Chroma supports metadata filtering via 'where' param in similarity_search if using .similarity_search_with_score
        # LangChain community's Chroma exposes 'similarity_search' with 'k' and 'filter' (where) in new versions.
        try:
            results = self.vs.similarity_search(query, k=k, filter=filter)  # type: ignore[arg-type]
        except TypeError:
            # Fallback for older versions without 'filter' param
            results = self.vs.similarity_search(query, k=k)

            # Manual filter if needed
            if filter:
                def match(meta, filt):
                    for kf, vf in filt.items():
                        if meta.get(kf) != vf:
                            return False
                    return True
                results = [r for r in results if match(r.metadata or {}, filter)]

        payload = []
        for i, r in enumerate(results, start=1):
            payload.append({
                "rank": i,
                "content": r.page_content,
                "metadata": r.metadata
            })
        return {
            "query": query,
            "results": payload
        }

    # ---------- Public accessors ----------
    def get_tools(self):
        """Tools for agents that need to STORE results."""
        return [self._store_tool]

    def get_retrieval_tools(self):
        """Tools for agents that need to RETRIEVE results (Q&A)."""
        return [self._retrieve_tool]
