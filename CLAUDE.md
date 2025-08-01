# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains examples of building DSPy RAG agents on Databricks that integrate with the Databricks Agent Framework, MLflow 3, MLflow Chat Agent, and deploy to Model Serving endpoints using `agents.deploy()`. The codebase is designed to run as interactive Databricks notebooks rather than traditional Python scripts.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Pipeline Execution](#pipeline-execution)
4. [Architecture](#architecture)
5. [MLflow Integration](#mlflow-integration)
6. [MLflow GenAI API](#mlflow-genai-api)
7. [MLflow Tracing API](#mlflow-tracing-api)
8. [Databricks Integration](#databricks-integration)
9. [Development Patterns](#development-patterns)
10. [Troubleshooting](#troubleshooting)

## Quick Start

### Required Dependencies
```python
%pip install --upgrade "mlflow[databricks]>=3.1" dspy-ai>=2.5 databricks-agents>=0.5 openai uv
dbutils.library.restartPython()
```

### Basic Usage
```python
from agent import DSPyRAGChatAgent

# Development configuration
agent = DSPyRAGChatAgent()  # Uses config.yaml

# Production configuration
optimized_config = mlflow.models.ModelConfig(development_config="config_optimized.yaml")
agent = DSPyRAGChatAgent(config=optimized_config)
```

## Environment Setup

### Required Environment Variables
```bash
export VS_INDEX_FULLNAME="catalog.schema.index_name"  # Vector Search index
export DSPY_LLM_ENDPOINT="databricks/databricks-claude-3-7-sonnet"  # Optional
export DSPY_OPTIMIZED_PROGRAM_PATH="/path/to/optimized/program.pkl"  # For optimized agents
```

### Local vs Databricks Environment Detection
```python
from dotenv import load_dotenv
import os
from databricks.connect import DatabricksSession

try:    
    load_dotenv()
    spark = DatabricksSession.builder.getOrCreate()
    
    if os.getenv('DATABRICKS_TOKEN') and os.getenv('DATABRICKS_HOST'):
        print("🏠 Local Development Mode Detected")
        user_name = "alex.miller@databricks.com"
    else:
        print("☁️  Databricks Environment Mode")
        user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
        
except ImportError:
    print("ℹ️  python-dotenv not available - using Databricks environment")
```

## Pipeline Execution

Execute notebooks in this order:
1. `dspy/rag-agent/01-dspy-data-preparation.py` - Creates Vector Search index
2. `dspy/rag-agent/02-create-eval-dataset.py` - Creates evaluation dataset
3. `dspy/rag-agent/02-dspy-pyfunc-rag-agent.py` - Creates basic RAG agent
4. `dspy/rag-agent/03-compile-optimized-rag.py` - Optimizes agent using DSPy
5. `dspy/rag-agent/04-deploy-optimized-agent.py` - Deploys to Model Serving

Optional: Run as Python scripts
```bash
python dspy/rag-agent/02-dspy-pyfunc-rag-agent.py
```

## Architecture

### Core Components
- **agent.py**: DSPyRAGChatAgent implementing MLflow ChatAgent interface
- **utils.py**: Helper functions for retrievers and optimization
- **config.yaml**: Development configuration
- **config_optimized.yaml**: Production configuration

### Configuration Schema
```yaml
llm_config:
  endpoint: "databricks/databricks-claude-3-7-sonnet"
  max_tokens: 2500  # 1500 for production
  temperature: 0.01  # 0.0 for production

vector_search:
  index_fullname: "catalog.schema.index_name"
  top_k: 5  # 3 for production

agent_config:
  use_optimized: true
  enable_tracing: true  # false for production
```

### MLflow ChatAgent Implementation
```python
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk, ChatAgentMessage, 
    ChatAgentResponse, ChatContext
)

class DSPyRAGChatAgent(ChatAgent):
    """MLflow ChatAgent that answers questions using a DSPy RAG program."""
    
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        latest_question = self._latest_user_prompt(messages)
        answer: str = self.rag(request=latest_question).response
        
        return ChatAgentResponse(messages=[
            ChatAgentMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content=answer,
            )
        ])
```

## MLflow Integration

### Model Logging and Registration
```python
with mlflow.start_run(run_name="optimized_dspy_rag_agent"):
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model="agent.py",
        model_config=model_config.to_dict(),
        artifacts={"optimized_program": optimized_program_path},
        pip_requirements=[
            f"dspy=={dspy.__version__}",
            f"databricks-agents>={agents_version}",
            f"mlflow=={mlflow.__version__}",
        ],
        resources=[
            DatabricksServingEndpoint(endpoint_name=llm_endpoint),
            DatabricksVectorSearchIndex(index_name=vs_index),
        ],
        input_example={"messages": [{"role": "user", "content": "Who is Zeus?"}]},
        code_paths=["agent.py", "utils.py"]
    )
    
    # Log metrics and parameters
    mlflow.log_param("optimization_method", "MIPROv2")
    mlflow.log_param("llm_endpoint", llm_config.get("endpoint"))
    mlflow.log_dict(model_config.to_dict(), "deployment_config.json")
```

### DSPy Integration
```python
# Enable DSPy autologging
mlflow.dspy.autolog()

# Configure DSPy with MLflow tracking
_lm = dspy.LM(
    endpoint,
    cache=False,
    max_tokens=max_tokens,
    temperature=temperature
)
```

## MLflow GenAI API

The MLflow GenAI API provides tools for working with generative AI models. Key features:

### Chat Completions
```python
import mlflow.genai

# Basic chat completion
response = mlflow.genai.chat.completions.create(
    messages=[{"role": "user", "content": "What is MLflow?"}],
    model="databricks-dbrx-instruct",
    temperature=0.7,
    max_tokens=1000
)

# Streaming responses
for chunk in mlflow.genai.chat.completions.create(
    messages=[{"role": "user", "content": "Explain RAG"}],
    model="databricks-llama-3-70b-instruct",
    stream=True
):
    print(chunk.choices[0].delta.content, end="")
```

### Embeddings
```python
# Generate embeddings
embeddings = mlflow.genai.embeddings.create(
    input=["Hello world", "MLflow is great"],
    model="databricks-bge-large-en"
)

# Access embedding vectors
vectors = [e.embedding for e in embeddings.data]
```

### Model Gateway Integration
```python
# Set gateway URI for GenAI
mlflow.genai.set_gateway_uri("databricks")

# List available endpoints
endpoints = mlflow.genai.list_endpoints()

# Get endpoint details
endpoint = mlflow.genai.get_endpoint("databricks-dbrx-instruct")
```

## MLflow Tracing API

MLflow Tracing provides comprehensive observability for LLM applications:

### Basic Tracing
```python
import mlflow
from mlflow.entities import SpanType

# Enable tracing
mlflow.set_experiment("dspy-rag-tracing")

# Trace a function
@mlflow.trace(span_type=SpanType.AGENT)
def rag_pipeline(query: str) -> str:
    with mlflow.start_span(name="retrieval", span_type=SpanType.RETRIEVER) as span:
        span.set_inputs({"query": query})
        docs = retriever.retrieve(query)
        span.set_outputs({"num_docs": len(docs)})
    
    with mlflow.start_span(name="generation", span_type=SpanType.LLM) as span:
        span.set_inputs({"query": query, "context": docs})
        response = llm.generate(query, docs)
        span.set_outputs({"response": response})
        span.set_attribute("model", "claude-3-sonnet")
    
    return response
```

### Automatic Tracing
```python
# Enable automatic tracing for supported libraries
mlflow.langchain.autolog()
mlflow.openai.autolog()
mlflow.dspy.autolog()  # For DSPy framework

# Disable tracing in production
mlflow.autolog(disable=True)
```

### Trace Analysis
```python
# Get trace information
with mlflow.start_run() as run:
    result = rag_pipeline("What is MLflow?")
    
    # Access trace data
    traces = mlflow.get_traces(run.info.run_id)
    for trace in traces:
        print(f"Trace ID: {trace.trace_id}")
        print(f"Duration: {trace.execution_time_ms}ms")
        
        # Analyze spans
        for span in trace.spans:
            print(f"  Span: {span.name} - {span.span_type}")
            print(f"  Duration: {span.end_time - span.start_time}ms")
```

### Custom Span Attributes
```python
@mlflow.trace
def process_with_metadata(text: str) -> dict:
    span = mlflow.get_current_active_span()
    
    # Set custom attributes
    span.set_attribute("text_length", len(text))
    span.set_attribute("processing_version", "2.0")
    
    # Set status
    span.set_status("OK")
    
    # Add events
    span.add_event("preprocessing_complete")
    
    result = {"processed": text.upper()}
    return result
```

### Tracing Configuration
```python
# Configure tracing behavior
mlflow.tracing.set_verbosity("DEBUG")  # or "INFO", "WARNING", "ERROR"

# Set sampling rate (0.0 to 1.0)
mlflow.tracing.set_sampling_rate(0.1)  # Sample 10% of traces

# Export traces to external systems
mlflow.tracing.export_traces(
    export_to="otlp",
    endpoint="http://localhost:4317"
)
```

## Databricks Integration

### Databricks-Specific Commands
```python
# Display rich formatted output
display(df)

# Create widgets for parameters
dbutils.widgets.text("vs_index_fullname", defaultValue="catalog.schema.index_name")
index_fullname = dbutils.widgets.get("vs_index_fullname")

# Access notebook context
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
user_name = context.userName().get()
workspace_url = context.browserHostName().get()
```

### Model Deployment
```python
from databricks import agents

# Deploy to Model Serving
deployment = agents.deploy(
    model_name="catalog.schema.model_name",
    model_version="1",
    scale_to_zero=True,
    environment_vars={
        "VS_INDEX_FULLNAME": index_fullname,
        "DSPY_OPTIMIZED_PROGRAM_PATH": optimized_path
    }
)

# Get deployment status
print(f"Endpoint: {deployment.endpoint_name}")
print(f"Status: {deployment.state}")
```

### Unity Catalog Integration
```python
# Set Unity Catalog as registry
mlflow.set_registry_uri("databricks-uc")

# Register model with UC
mlflow.register_model(
    model_uri=model_info.model_uri,
    name="catalog.schema.model_name"
)
```

## Development Patterns

### DSPy Program Structure
```python
class _DSPyRAGProgram(dspy.Module):
    def __init__(self, retriever: DatabricksRM):
        super().__init__()
        self.retriever = retriever
        self.response_generator = dspy.ChainOfThought("context, request -> response")
    
    def forward(self, request: str):
        retrieved_context = self.retriever(request)
        with dspy.context(lm=_lm):
            response = self.response_generator(
                context=retrieved_context.docs, 
                request=request
            ).response
        return dspy.Prediction(response=response)
```

### Optimization Loading
```python
def load_optimized_program(program_class, config):
    """Load optimized DSPy program from Unity Catalog volume."""
    path = config.get("agent_config", {}).get("optimized_program_path")
    if path and os.path.exists(path):
        return dspy.teleprompt.load_program(path)
    return None
```

### Vector Search Configuration
```python
retriever = DatabricksRM(
    databricks_index_name=index_fullname,
    databricks_endpoint=vs_endpoint,
    top_k=top_k,
    columns=["page_content", "url", "metadata"],
    filters={"source": "wikipedia"}  # Optional filtering
)
```

## Troubleshooting

### Common Issues
1. **Unity Catalog Permissions**: Ensure CREATE, READ, WRITE on catalog/schema
2. **Vector Search**: Verify index is ACTIVE and synced
3. **Environment Variables**: Check all required vars are set
4. **Model Serving**: Monitor endpoint logs for deployment issues

### Debug Commands
```python
# Check MLflow tracking
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Registry URI: {mlflow.get_registry_uri()}")

# Verify Vector Search
from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()
index = client.get_index(index_fullname)
print(f"Index Status: {index.status}")

# Test retriever
test_results = retriever("test query")
print(f"Retrieved {len(test_results.docs)} documents")
```

### Performance Optimization
- Use `config_optimized.yaml` for production
- Disable tracing in production: `enable_tracing: false`
- Reduce token limits and top_k for faster responses
- Enable model caching when appropriate
- Use scale-to-zero for cost optimization