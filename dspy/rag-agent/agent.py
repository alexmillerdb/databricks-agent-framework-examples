# DSPy RAG → MLflow ChatAgent wrapper
# This lightweight module is intended to be imported by notebooks or scripts that
# need a ready-to-use ChatAgent backed by DSPy and Databricks Vector Search.

from __future__ import annotations

import os
import uuid
from typing import Any, List, Optional

import dspy
import mlflow
from dspy.retrieve.databricks_rm import DatabricksRM
from mlflow.pyfunc import ChatAgent
from mlflow.entities import SpanType
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
)

# -----------------------------------------------------------------------------
# Global DSPy / MLflow configuration
# -----------------------------------------------------------------------------

# Enable tracing of every DSPy span to MLflow runs (compile, eval, inference).
mlflow.dspy.autolog()

# Configure the LLM endpoint to use for generation. Override via env var
LLM_ENDPOINT = os.getenv("DSPY_LLM_ENDPOINT", "databricks/databricks-claude-3-7-sonnet")

# Instantiate the LM and set DSPy global settings
_lm = dspy.LM(model=LLM_ENDPOINT)
dspy.settings.configure(lm=_lm)

# -----------------------------------------------------------------------------
# Helper: build a Databricks Vector Search retriever
# -----------------------------------------------------------------------------


def _build_retriever() -> DatabricksRM:
    """Create a retriever for the Vector Search index specified by env vars.

    Required env vars:
      * VS_INDEX_FULLNAME  – catalog.schema.index_name (Unity Catalog path)
    """

    index_fullname = os.getenv("VS_INDEX_FULLNAME")
    if not index_fullname:
        raise ValueError(
            "Environment variable VS_INDEX_FULLNAME must be set to the fully-qualified "
            "Vector Search index (e.g. catalog.schema.index_name)"
        )

    return DatabricksRM(
        databricks_index_name=index_fullname,
        text_column_name="chunk",
        docs_id_column_name="id",
        columns=["id", "title", "chunk_id"],
        k=int(os.getenv("DSPY_TOP_K", "5")),
    )


# -----------------------------------------------------------------------------
# Define the DSPy RAG program (very similar to dspy-create-rag-program.py)
# -----------------------------------------------------------------------------

_RESPONSE_GENERATOR_SIGNATURE = "context, request -> response"


class _DSPyRAGProgram(dspy.Module):
    """Minimal DSPy module: retrieve → generate."""

    def __init__(self, retriever: DatabricksRM):
        super().__init__()
        self.retriever = retriever
        self.lm = _lm
        self.response_generator = dspy.ChainOfThought(_RESPONSE_GENERATOR_SIGNATURE)

    def forward(self, request: str):
        retrieved_context = self.retriever(request)
        with dspy.context(lm=self.lm):
            response = self.response_generator(
                context=retrieved_context.docs, request=request
            ).response
        return dspy.Prediction(response=response)


# -----------------------------------------------------------------------------
# MLflow ChatAgent implementation
# -----------------------------------------------------------------------------


class DSPyRAGChatAgent(ChatAgent):
    """MLflow ChatAgent that answers questions using a DSPy RAG program."""

    def __init__(self, rag_program: Optional[_DSPyRAGProgram] = None):
        super().__init__()
        # If caller provides a custom program, use it, else build default
        self.rag = rag_program or _DSPyRAGProgram(_build_retriever())

    # -------------------------------------------------- internal helpers ----
    @staticmethod
    def _latest_user_prompt(messages: List[Any]) -> str:
        if not messages:
            return ""
        last = messages[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return getattr(last, "content", "")

    def prepare_message_history(self, messages: list[ChatAgentMessage]):
        history_entries = []
        # Assume the last message in the input is the most recent user question.
        for i in range(0, len(messages) - 1, 2):
            history_entries.append({"question": messages[i].content, "answer": messages[i + 1].content})
        return dspy.History(messages=history_entries)

    # ------------------------------------------------------ required API ----
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        latest_question = self._latest_user_prompt(messages)
        answer: str = self.rag(question=latest_question, history=self.prepare_message_history(messages)).response

        assistant_msg = ChatAgentMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=answer,
        )
        return ChatAgentResponse(messages=[assistant_msg])

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ):
        latest_question = self._latest_user_prompt(messages)
        answer: str = self.rag(question=latest_question, history=self.prepare_message_history(messages)).response

        yield ChatAgentChunk(
            delta=ChatAgentMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content=answer,
            )
        )


# -----------------------------------------------------------------------------
# Convenience: expose an AGENT instance that mlflow.models.set_model can pick up
# -----------------------------------------------------------------------------

from mlflow.models import set_model

AGENT = DSPyRAGChatAgent()
set_model(AGENT)