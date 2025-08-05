# DSPy RAG → MLflow ChatAgent wrapper
# This lightweight module is intended to be imported by notebooks or scripts that
# need a ready-to-use ChatAgent backed by DSPy and Databricks Vector Search.

# Configuration Usage Examples:
# 
# Development/Default usage:
#   agent = DSPyRAGChatAgent()  # Uses config.yaml
#
# Production/Optimized usage:
#   optimized_config = mlflow.models.ModelConfig(development_config="config_optimized.yaml")
#   agent = DSPyRAGChatAgent(config=optimized_config)
#
# Custom configuration:
#   custom_config = mlflow.models.ModelConfig(development_config="my_custom_config.yaml")
#   agent = DSPyRAGChatAgent(config=custom_config)

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
    ChatContext,
)

from modules.utils import build_retriever, load_optimized_program

# -----------------------------------------------------------------------------
# Configuration Management
# -----------------------------------------------------------------------------

# Load configuration from config.yaml (default development config)
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
model_config = mlflow.models.ModelConfig(development_config=CONFIG_FILE)

# -----------------------------------------------------------------------------
# Global DSPy / MLflow configuration
# -----------------------------------------------------------------------------

# Get MLflow configuration
mlflow_config = model_config.get("mlflow_config") or {}
if mlflow_config.get("enable_autolog", True):
    # Enable tracing of every DSPy span to MLflow runs (compile, eval, inference).
    mlflow.dspy.autolog()

# Get LLM configuration
llm_config = model_config.get("llm_config") or {}
llm_endpoints = model_config.get("llm_endpoints") or {}

# Helper function to create LLM instances for different components
def create_llm_for_component(component_name: str, fallback_config: dict = None):
    """Create an LLM instance for a specific component.
    
    Args:
        component_name: Name of the component (e.g., 'query_rewriter', 'response_generator')
        fallback_config: Fallback configuration if component-specific config not found
    
    Returns:
        dspy.LM instance configured for the component
    """
    # Use component-specific config if available, otherwise fallback to default
    component_config = llm_endpoints.get(component_name, fallback_config or llm_config)
    
    # Allow environment variable override for each component
    env_var_name = f"DSPY_{component_name.upper()}_ENDPOINT"
    endpoint = os.getenv(env_var_name, component_config.get("endpoint", "databricks/databricks-claude-3-7-sonnet"))
    
    return dspy.LM(
        model=endpoint,
        max_tokens=component_config.get("max_tokens", 2500),
        temperature=component_config.get("temperature", 0.01),
        top_p=component_config.get("top_p", 0.95)
    )

# Create default LLM (for backward compatibility)
LLM_ENDPOINT = os.getenv("DSPY_LLM_ENDPOINT", llm_config.get("endpoint", "databricks/databricks-claude-3-7-sonnet"))
_lm = dspy.LM(
    model=LLM_ENDPOINT,
    max_tokens=llm_config.get("max_tokens", 2500),
    temperature=llm_config.get("temperature", 0.01),
    top_p=llm_config.get("top_p", 0.95)
)
dspy.settings.configure(lm=_lm)

# -----------------------------------------------------------------------------
# Define the DSPy RAG program (very similar to dspy-create-rag-program.py)
# -----------------------------------------------------------------------------

# Get DSPy configuration
dspy_config = model_config.get("dspy_config") or {}
_RESPONSE_GENERATOR_SIGNATURE = dspy_config.get("response_generator_signature", "context, request -> response")

class RewriteQuery(dspy.Signature):
    """
    Rewrite the user's question to optimize it for vector search retrieval.
    Make the query more specific, add relevant keywords, and ensure it captures the user's intent.
    """
    original_question = dspy.InputField(desc="The original user question")
    rewritten_query = dspy.OutputField(desc="Optimized query for better retrieval results")

class GenerateCitedAnswer(dspy.Signature):
    """
    Answer the user's question using only the information from the provided context passages.
    When using information from a passage, cite it with [n], where n is the passage number as provided.
    For example: "Air pollution can affect respiratory health [1]."
    Only use the passages as sources for your answer.
    """
    context = dspy.InputField(desc="List of context passages, each assigned a number starting from 1")
    question = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="Concise answer that cites source passages, e.g., [1], [2]")


class _DSPyRAGProgram(dspy.Module):
    """Minimal DSPy module: retrieve → generate."""

    def __init__(self, retriever: DatabricksRM, config=None):
        super().__init__()
        self.retriever = retriever
        self.config = config or model_config
        
        # Create component-specific LLMs
        self.query_rewriter_lm = create_llm_for_component("query_rewriter")
        self.response_generator_lm = create_llm_for_component("response_generator")
        
        # Use default LM as fallback
        self.lm = _lm
        
        # Initialize response generator with its specific LLM
        self.response_generator = dspy.ChainOfThought(GenerateCitedAnswer)
        
        # Get field names from config
        vs_config = self.config.get("vector_search") or {}
        agent_config = self.config.get("agent_config") or {}
        
        # Query rewriting configuration
        self.use_query_rewriter = agent_config.get("use_query_rewriter", True)
        if self.use_query_rewriter:
            self.query_rewriter = dspy.ChainOfThought(RewriteQuery)
        
        # Dynamic field mapping from config
        self.text_field = vs_config.get("text_column_name", "chunk")
        self.id_field = vs_config.get("docs_id_column_name", "id")
        self.columns = vs_config.get("columns", ["id", "title", "chunk_id"])
        
        # Extract metadata fields (exclude the main text and id fields)
        self.metadata_fields = [col for col in self.columns if col not in [self.text_field, self.id_field]]

    def forward(self, request: str):

        # Optionally rewrite the query for better retrieval
        search_query = request
        if self.use_query_rewriter:
            with dspy.context(lm=self.query_rewriter_lm):
                rewrite_result = self.query_rewriter(original_question=request)
                search_query = rewrite_result.rewritten_query
                print(f"🔄 Original: {request}")
                print(f"🎯 Rewritten: {search_query}")

        # Get the context from the retriever using the (possibly rewritten) query
        retrieved_context = self.retriever(search_query)
        
        # Handle DSPy Prediction objects from retriever
        if hasattr(retrieved_context, 'docs'):
            # DSPy retriever returned a Prediction with docs attribute
            documents = retrieved_context.docs
            print(f"🔍 Retrieved {len(documents) if documents else 0} documents from Prediction.docs")
        elif hasattr(retrieved_context, '_store') and 'docs' in retrieved_context._store:
            # Access docs from internal store
            documents = retrieved_context._store['docs']
            print(f"🔍 Retrieved {len(documents) if documents else 0} documents from Prediction._store['docs']")
        elif isinstance(retrieved_context, list):
            # Direct list of documents
            documents = retrieved_context
            print(f"🔍 Retrieved {len(documents) if documents else 0} documents as list")
        else:
            # Fallback - try to convert to list
            try:
                documents = list(retrieved_context)
                print(f"🔍 Retrieved {len(documents) if documents else 0} documents via list conversion")
            except:
                print(f"❌ Cannot extract documents from retriever response of type {type(retrieved_context)}")
                if hasattr(retrieved_context, '__dict__'):
                    print(f"📋 Available attributes: {list(retrieved_context.__dict__.keys())}")
                return dspy.Prediction(response="I encountered an error accessing the retrieved documents.")
        
        # Validate we have documents
        if not documents or len(documents) == 0:
            print("⚠️  No documents retrieved - vector search returned empty results")
            return dspy.Prediction(response="I couldn't find any relevant information to answer your question.")
        
        # Debug: show structure of first document
        try:
            first_doc = documents[0]
            print(f"📄 Sample document type: {type(first_doc)}")
            if hasattr(first_doc, '__dict__'):
                available_fields = list(first_doc.__dict__.keys()) if hasattr(first_doc, '__dict__') else []
                print(f"📋 Available fields: {available_fields}")
            elif isinstance(first_doc, dict):
                print(f"📋 Available keys: {list(first_doc.keys())}")
        except (IndexError, AttributeError) as e:
            print(f"⚠️  Error accessing first document: {e}")

        # Format passages with numbers and metadata for better context
        numbered_passages = []
        content_extraction_issues = []
        
        for i, passage in enumerate(documents):
            content = None
            metadata_info = ""
            extraction_method = "unknown"
            
            # Handle structured passage objects
            if hasattr(passage, self.text_field):
                # Vector search returns objects with configured text field
                content = getattr(passage, self.text_field)
                extraction_method = f"attribute.{self.text_field}"
                
                # Add metadata from all configured columns
                for field in self.metadata_fields:
                    if hasattr(passage, field):
                        value = getattr(passage, field)
                        if value:
                            metadata_info += f" ({field.title()}: {value})"
                
                # Add ID if available and different from text field
                if hasattr(passage, self.id_field):
                    id_value = getattr(passage, self.id_field)
                    if id_value:
                        metadata_info += f" ({self.id_field.upper()}: {id_value})"
                        
            elif isinstance(passage, dict):
                # Handle dictionary format - use configured field names
                content = passage.get(self.text_field)
                extraction_method = f"dict.{self.text_field}"
                
                if not content:
                    # Fallback to common field names
                    for fallback_field in ['content', 'text', 'chunk']:
                        if fallback_field in passage and passage[fallback_field]:
                            content = passage[fallback_field]
                            extraction_method = f"dict.{fallback_field}_fallback"
                            break
                    
                    if not content:
                        content = str(passage)
                        extraction_method = "dict.str_fallback"
                
                # Add metadata from configured columns
                for field in self.metadata_fields:
                    value = passage.get(field)
                    if value:
                        metadata_info += f" ({field.title()}: {value})"
                
                # Add ID if available
                id_value = passage.get(self.id_field)
                if id_value:
                    metadata_info += f" ({self.id_field.upper()}: {id_value})"
            else:
                # Fallback for simple string passages
                content = str(passage)
                extraction_method = "str_fallback"
            
            # Validate content quality
            if not content or (isinstance(content, str) and len(content.strip()) == 0):
                content_extraction_issues.append(f"Passage {i+1}: Empty content via {extraction_method}")
                content = f"[No content available from passage {i+1}]"
                
            elif isinstance(content, str) and len(content.strip()) < 10:
                content_extraction_issues.append(f"Passage {i+1}: Very short content ({len(content.strip())} chars) via {extraction_method}")
            
            # Debug: show content extraction for first few passages
            if i < 2:  # Only show first 2 to avoid spam
                content_preview = content[:100] + "..." if len(str(content)) > 100 else str(content)
                print(f"📄 Passage {i+1} extracted via {extraction_method}: {content_preview}")
            
            # Add numbered passage
            numbered_passages.append(f"[{i+1}] {content}{metadata_info}")
        
        # Report content extraction issues
        if content_extraction_issues:
            print(f"⚠️  Content extraction issues: {'; '.join(content_extraction_issues)}")
        
        # Validate we have meaningful content
        total_content_length = sum(len(str(p)) for p in numbered_passages)
        print(f"📊 Total context length: {total_content_length} characters across {len(numbered_passages)} passages")
        
        if total_content_length < 50:  # Very little content
            print("⚠️  Very little content retrieved - responses may be poor")
        
        context = "\n\n".join(numbered_passages)

        # Generate the response with the response generator's specific LLM
        with dspy.context(lm=self.response_generator_lm):
            result = self.response_generator(
                context=context, question=request
            )
            response = result.answer  # Use 'answer' from GenerateCitedAnswer signature
        return dspy.Prediction(response=response)


# -----------------------------------------------------------------------------
# MLflow ChatAgent implementation
# -----------------------------------------------------------------------------


class DSPyRAGChatAgent(ChatAgent):
    """MLflow ChatAgent that answers questions using a DSPy RAG program."""

    def __init__(self, rag_program: Optional[_DSPyRAGProgram] = None, config=None):
        super().__init__()
        
        # Use provided config or global model_config
        self.config = config or model_config
        self.rag_program = rag_program  # Store for later use in load_context
        self.rag = None  # Will be set in load_context or here
        
        # If we have a custom program, use it immediately
        if rag_program:
            self.rag = rag_program
        else:
            # Otherwise, we'll set up the program in load_context (for MLflow deployment)
            # or here (for direct usage)
            self._setup_rag_program()

    def load_context(self, context):
        """
        MLflow calls this method when loading the model with artifacts.
        This is where we should load the optimized program from artifacts.
        """
        print("🔄 Loading MLflow context...")
        
        # Store the MLflow context for artifact access
        self.context = context
        
        # Try to load optimized program from artifacts first
        if hasattr(context, 'artifacts') and context.artifacts:
            print(f"📦 Available artifacts: {list(context.artifacts.keys())}")
            
            if "optimized_program" in context.artifacts:
                artifact_path = context.artifacts["optimized_program"]
                print(f"🎯 Loading optimized program from artifact: {artifact_path}")
                
                try:
                    optimized_program = self._load_optimized_from_artifact(artifact_path)
                    if optimized_program:
                        self.rag = optimized_program
                        print("✅ Successfully loaded optimized program from artifact")
                        return
                except Exception as e:
                    print(f"⚠️  Failed to load optimized program from artifact: {e}")
        
        # Fallback to regular setup if artifact loading fails
        print("🔄 Falling back to regular program setup...")
        self._setup_rag_program()

    def _load_optimized_from_artifact(self, artifact_path: str) -> Optional[_DSPyRAGProgram]:
        """
        Load optimized DSPy program from MLflow artifact.
        
        Args:
            artifact_path: Path to the optimized program artifact
            
        Returns:
            Loaded DSPy program or None if loading fails
        """
        if not os.path.exists(artifact_path):
            print(f"❌ Artifact path does not exist: {artifact_path}")
            return None
        
        try:
            # Create a base program with retriever and config
            base_program = _DSPyRAGProgram(build_retriever(self.config), self.config)
            
            # Load the optimized state
            if artifact_path.endswith('.json'):
                base_program.load(artifact_path)
                return base_program
            else:
                # Handle pickle files (fallback)
                import pickle
                with open(artifact_path, 'rb') as f:
                    loaded_program = pickle.load(f)
                    # Ensure the loaded program has the correct LLMs
                    if hasattr(loaded_program, 'query_rewriter_lm'):
                        loaded_program.query_rewriter_lm = create_llm_for_component("query_rewriter")
                    if hasattr(loaded_program, 'response_generator_lm'):
                        loaded_program.response_generator_lm = create_llm_for_component("response_generator")
                    return loaded_program
                    
        except Exception as e:
            print(f"❌ Error loading optimized program: {e}")
            return None

    def _setup_rag_program(self):
        """
        Set up the RAG program using configuration or fallback methods.
        This handles both MLflow deployment and direct usage scenarios.
        """
        agent_config = self.config.get("agent_config") or {}
        use_optimized = agent_config.get("use_optimized", True)
        
        if use_optimized:
            # Try to load optimized program using various methods
            optimized = load_optimized_program(_DSPyRAGProgram, self.config, getattr(self, 'context', None))
            if optimized:
                print("✅ Using optimized DSPy program")
                self.rag = optimized
            else:
                print("⚠️  No optimized program found, using default")
                self.rag = _DSPyRAGProgram(build_retriever(self.config), self.config)
        else:
            print("🔄 Using base DSPy program (optimization disabled)")
            self.rag = _DSPyRAGProgram(build_retriever(self.config), self.config)

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
        # Convert dict messages to ChatAgentMessage objects if needed
        if messages and isinstance(messages[0], dict):
            messages = [ChatAgentMessage(role=msg["role"], content=msg["content"]) for msg in messages]
        
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
        # Convert dict messages to ChatAgentMessage objects if needed
        if messages and isinstance(messages[0], dict):
            messages = [ChatAgentMessage(role=msg["role"], content=msg["content"]) for msg in messages]
            
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