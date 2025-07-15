# DSPy RAG → MLflow ChatAgent wrapper
# This lightweight module is intended to be imported by notebooks or scripts that
# need a ready-to-use ChatAgent backed by DSPy and Databricks Vector Search.

from __future__ import annotations

import os
import uuid
import json
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
# Helper: load optimized program if available
# -----------------------------------------------------------------------------

def _load_optimized_program() -> Optional[_DSPyRAGProgram]:
    """Load a pre-compiled optimized DSPy program if available."""
    optimized_path = os.getenv("DSPY_OPTIMIZED_PROGRAM_PATH")
    
    if optimized_path and os.path.exists(optimized_path):
        try:
            # Try DSPy's native load method first
            if optimized_path.endswith('.json'):
                # Load using DSPy's save/load mechanism
                program = _DSPyRAGProgram(_build_retriever())
                program.load(optimized_path)
                return program
            else:
                # Fallback to pickle (less reliable)
                import pickle
                with open(optimized_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load optimized program: {e}")
            
            # Try loading components
            components_path = optimized_path.replace('.pkl', '_components.json')
            if os.path.exists(components_path):
                try:
                    return _load_from_components(components_path)
                except Exception as e2:
                    print(f"Failed to load from components: {e2}")
    
    return None

def _load_from_components(components_path: str) -> _DSPyRAGProgram:
    """Load program from saved components."""
    with open(components_path, 'r') as f:
        components = json.load(f)
    
    # Create base program
    retriever = _build_retriever()
    program = _DSPyRAGProgram(retriever)
    
    # Apply optimized components
    if 'signature' in components and 'instructions' in components['signature']:
        program.response_generator.signature.instructions = components['signature']['instructions']
    
    return program

# -----------------------------------------------------------------------------
# MLflow ChatAgent implementation
# -----------------------------------------------------------------------------


class DSPyRAGChatAgent(ChatAgent):
    """MLflow ChatAgent that answers questions using a DSPy RAG program."""

    def __init__(self, rag_program: Optional[_DSPyRAGProgram] = None, use_optimized: bool = True):
        super().__init__()
        
        # Priority: custom program > optimized program > default program
        if rag_program:
            self.rag = rag_program
        elif use_optimized:
            optimized = _load_optimized_program()
            if optimized:
                print("Using optimized DSPy program")
                self.rag = optimized
            else:
                print("No optimized program found, using default")
                self.rag = _DSPyRAGProgram(_build_retriever())
        else:
            self.rag = _DSPyRAGProgram(_build_retriever())

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
        answer: str = self.rag(request=latest_question).response

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
        answer: str = self.rag(request=latest_question).response

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