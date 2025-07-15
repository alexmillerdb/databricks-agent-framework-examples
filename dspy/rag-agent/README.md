# DSPy RAG Agent Framework

A comprehensive framework for building, optimizing, and deploying Retrieval-Augmented Generation (RAG) agents using DSPy and Databricks. This framework provides a complete end-to-end pipeline from data preparation to production deployment.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)
7. [Optimization](#optimization)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)

## Overview

This framework demonstrates how to build production-ready RAG agents using interactive Databricks notebooks. The framework uses:

- **DSPy**: For building and optimizing language model programs
- **Databricks Vector Search**: For efficient document retrieval
- **MLflow**: For experiment tracking, model management, and deployment
- **MLflow ChatAgent**: For standardized agent interfaces

> **Note**: All components are designed as interactive Databricks notebooks that you run cell-by-cell in your Databricks workspace. Each notebook includes detailed markdown documentation and step-by-step instructions.

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Prep     │───▶│   Vector Search │───▶│   RAG Agent     │
│   (Step 1)      │    │   Index (Step 1)│    │   (Step 2)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Deployment    │◀───│   Optimization  │◀───│   Evaluation    │
│   (Step 4)      │    │   (Step 3)      │    │   (Step 3)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### Environment Requirements

- **Databricks Runtime**: Serverless CPU with version 3
- **Databricks Workspace**: With Unity Catalog enabled
- **Vector Search Endpoint**: Pre-configured Databricks Vector Search endpoint

### Python Dependencies

Install in Databricks notebook cells:

```python
%pip install --upgrade "mlflow[databricks]>=3.1" dspy-ai>=2.5 databricks-agents>=0.5 openai uv
dbutils.library.restartPython()
```

### Required Permissions

- Unity Catalog: `CREATE`, `READ`, `WRITE` permissions on target catalog/schema
- Vector Search: Access to create and query vector search indexes
- Model Serving: Permissions to deploy models to serving endpoints

## Quick Start

### 1. Import Notebooks to Databricks

1. Upload the notebook files to your Databricks workspace
2. Or clone the repository and import the `dspy/rag-agent/` folder

### 2. Configure Environment Variables

Set these variables in your Databricks workspace or in the notebook cells:

```python
# Set in Databricks workspace environment or notebook
import os
os.environ["VS_INDEX_FULLNAME"] = "your_catalog.your_schema.your_index_name"
os.environ["DSPY_LLM_ENDPOINT"] = "databricks/databricks-claude-3-7-sonnet"
```

### 3. Run the Complete Pipeline

Open and run the notebooks in order in your Databricks workspace:

1. **Step 1**: `01-dspy-data-preparation.py` - Data preparation and vector search index creation
2. **Step 2**: `02-dspy-pyfunc-rag-agent.py` - Create basic RAG agent  
3. **Step 3**: `03-compile-optimized-rag.py` - Optimize the agent (optional but recommended)
4. **Step 4**: `04-deploy-optimized-agent.py` - Deploy optimized agent

Each notebook contains interactive cells that you can run sequentially.

## Step-by-Step Guide

### Step 1: Data Preparation and Vector Search Index

**Notebook**: `01-dspy-data-preparation.py`

This interactive notebook prepares your data and creates a Vector Search index for retrieval.

#### What it does:
- Loads source data (Wikipedia articles in this example)
- Chunks the text using `RecursiveCharacterTextSplitter`
- Creates a Delta table with chunked documents
- Creates a Vector Search index with embeddings

#### Key configurations:
The notebook uses Databricks widgets for configuration. Update these values in the widget interface:
```python
# Databricks widgets (configure in the notebook interface)
source_catalog = "users"
source_schema = "your_username"
source_table = "wikipedia_chunks"
vs_endpoint = "your-vector-search-endpoint"
vs_index = "wikipedia_chunks_index"
```

#### Running the notebook:
1. Open `01-dspy-data-preparation.py` in your Databricks workspace
2. Configure the widget values at the top of the notebook
3. Run all cells sequentially using **Run All** or execute cells individually

#### Expected output:
- Delta table: `{catalog}.{schema}.{table}` with chunked documents
- Vector Search index: `{catalog}.{schema}.{index}` ready for querying

### Step 2: Create Basic RAG Agent

**Notebook**: `02-dspy-pyfunc-rag-agent.py`

This interactive notebook creates a basic RAG agent using DSPy and wraps it with MLflow ChatAgent.

#### What it does:
- Sets up DSPy with your chosen LLM endpoint
- Creates a retriever using Databricks Vector Search
- Implements a DSPy RAG program with retrieval and generation
- Wraps the program in MLflow ChatAgent for standardized interface
- Logs the agent as an MLflow model

#### Key features:
- **MLflow tracing**: Full DSPy span capture
- **Configuration management**: Using MLflow ModelConfig
- **Deployment ready**: Can be deployed to Model Serving

#### Running the notebook:
1. Open `02-dspy-pyfunc-rag-agent.py` in your Databricks workspace
2. Ensure environment variables are set (or update the variables in the notebook)
3. Run all cells sequentially to build and test the RAG agent
4. The final cells will log and deploy the model

#### Expected output:
- Logged MLflow model ready for deployment
- Registered model in Unity Catalog
- Deployed endpoint (if deployment is enabled)

### Step 3: Optimize the RAG Agent

**Notebook**: `03-compile-optimized-rag.py`

This interactive notebook optimizes the RAG agent using DSPy's compilation techniques.

#### What it does:
- Creates training examples for optimization
- Defines custom evaluation metrics
- Uses DSPy optimizers (MIPROv2, BootstrapFewShot) to improve performance
- Evaluates base vs optimized program performance
- Saves the optimized program for deployment

#### Optimization strategies:
- **MIPROv2**: Advanced multi-stage instruction optimization
- **BootstrapFewShot**: Few-shot learning with bootstrapping
- **Custom metrics**: Domain-specific evaluation criteria

#### Running the notebook:
1. Open `03-compile-optimized-rag.py` in your Databricks workspace
2. Run the configuration cells to set up the optimization environment
3. Execute the training dataset creation cells
4. Run the optimization cells (this may take several minutes)
5. Review the performance comparison results

#### Expected output:
- Optimized DSPy program saved as artifacts
- Performance comparison metrics
- Optimized program ready for deployment

### Step 4: Deploy Optimized Agent

**Notebook**: `04-deploy-optimized-agent.py`

This interactive notebook deploys the optimized agent to Databricks Model Serving.

#### What it does:
- Downloads the optimized program from MLflow artifacts
- Creates an agent with optimized settings
- Logs the optimized agent as a new MLflow model
- Deploys to Model Serving with production configuration

#### Production features:
- **Optimized configuration**: Using `config_optimized.yaml`
- **Performance monitoring**: Built-in metrics and logging
- **Scalability**: Auto-scaling with scale-to-zero support

#### Running the notebook:
1. Open `04-deploy-optimized-agent.py` in your Databricks workspace
2. Run the configuration cells to load optimized settings
3. Execute the model download cells to retrieve the optimized program
4. Test the optimized agent with the provided test cells
5. Run the deployment cells to deploy to Model Serving

#### Expected output:
- Deployed model serving endpoint
- Production-ready agent with optimized performance

## Configuration

### Development Configuration (`config.yaml`)

```yaml
# LLM Configuration
llm_config:
  endpoint: "databricks/databricks-claude-3-7-sonnet"
  max_tokens: 2500
  temperature: 0.01
  top_p: 0.95

# Vector Search Configuration
vector_search:
  index_fullname: users.alex_miller.wikipedia_chunks_index
  text_column_name: "chunk"
  docs_id_column_name: "id"
  columns: ["id", "title", "chunk_id"]
  top_k: 5

# Agent Configuration
agent_config:
  use_optimized: true
  enable_tracing: true
  verbose: true
  max_iterations: 10

# MLflow Configuration
mlflow_config:
  enable_autolog: true
  experiment_name: "dspy-rag-agent"
  registered_model_name: "dspy_rag_chat_agent"
```

### Production Configuration (`config_optimized.yaml`)

```yaml
# LLM Configuration - Optimized for performance
llm_config:
  endpoint: "databricks/databricks-claude-3-7-sonnet"
  max_tokens: 1500  # Reduced for faster response times
  temperature: 0.0  # More deterministic responses
  top_p: 0.9

# Vector Search Configuration - Optimized retrieval
vector_search:
  top_k: 3  # Reduced for faster retrieval and more focused context

# Agent Configuration - Optimized for production
agent_config:
  use_optimized: true
  enable_tracing: false  # Disabled for performance in production
  verbose: false
  max_iterations: 5
```

### Environment Variables

```bash
# Required
export VS_INDEX_FULLNAME="catalog.schema.index_name"

# Optional overrides
export DSPY_LLM_ENDPOINT="databricks/your-endpoint"
export DSPY_TOP_K="5"
export DSPY_OPTIMIZED_PROGRAM_PATH="/path/to/optimized/program.pkl"
```

## Usage Examples

### Basic Usage

```python
from agent import DSPyRAGChatAgent
from mlflow.types.agent import ChatAgentMessage

# Create agent with default configuration
agent = DSPyRAGChatAgent()

# Ask a question
message = ChatAgentMessage(
    id="msg-1",
    role="user",
    content="What is machine learning?"
)

response = agent.predict([message])
print(response.messages[0].content)
```

### Using Custom Configuration

```python
import mlflow
from agent import DSPyRAGChatAgent

# Load custom configuration
config = mlflow.models.ModelConfig(
    development_config="config_optimized.yaml"
)

# Create agent with custom config
agent = DSPyRAGChatAgent(config=config)
```

### Programmatic Configuration

```python
import mlflow
from agent import DSPyRAGChatAgent

# Define custom configuration
custom_config = {
    "llm_config": {
        "endpoint": "databricks/databricks-claude-3-7-sonnet",
        "max_tokens": 1000,
        "temperature": 0.5
    },
    "vector_search": {
        "top_k": 4
    },
    "agent_config": {
        "use_optimized": False,
        "enable_tracing": True
    }
}

# Create ModelConfig from dictionary
config = mlflow.models.ModelConfig(development_config=custom_config)
agent = DSPyRAGChatAgent(config=config)
```

## Optimization

### DSPy Optimization Techniques

#### 1. MIPROv2 (Recommended)
- **Multi-stage optimization**: Iteratively improves instructions and demonstrations
- **Automatic mode selection**: Chooses appropriate optimization strategy
- **Performance tracking**: Monitors optimization progress

```python
optimizer = dspy.MIPROv2(
    metric=rag_evaluation_metric,
    auto="medium",
    num_threads=8,
    verbose=True,
    track_stats=True
)
```

#### 2. BootstrapFewShot
- **Few-shot learning**: Learns from provided examples
- **Bootstrapping**: Generates additional training examples
- **Fallback option**: When MIPROv2 fails

```python
optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=rag_evaluation_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
```

### Custom Evaluation Metrics

```python
def rag_evaluation_metric(gold, pred, trace=None):
    """Custom metric for evaluating RAG responses."""
    request = gold.request
    expected = gold.expected_response
    actual = pred.response
    
    # Use LLM-as-judge for evaluation
    judge = dspy.Predict(ResponseAssessment)(
        assessed_text=actual,
        assessment_question=f"Rate how well '{actual}' answers '{request}'"
    )
    
    return 1.0 if "correct" in judge.assessment_answer.lower() else 0.0
```

## Deployment

### Model Serving Deployment

```python
from databricks import agents

# Deploy with environment variables
deployment_config = {
    "DSPY_LLM_ENDPOINT": "databricks/databricks-claude-3-7-sonnet",
    "VS_INDEX_FULLNAME": "catalog.schema.index_name",
    "DSPY_OPTIMIZED_PROGRAM_PATH": "/path/to/optimized/program.pkl"
}

agents.deploy(
    model_name="catalog.schema.model_name",
    model_version="1",
    scale_to_zero=True,
    environment_vars=deployment_config
)
```

### Loading Deployed Models

```python
import mlflow

# Load model from registry
model = mlflow.pyfunc.load_model(
    model_uri="models:/catalog.schema.model_name/1"
)

# Use the model
result = model.predict({
    "messages": [{"role": "user", "content": "Your question here"}]
})
```

## Troubleshooting

### Common Issues

#### 1. Vector Search Index Not Found
```bash
Error: RESOURCE_DOES_NOT_EXIST
```
**Solution**: Ensure the Vector Search index exists and `VS_INDEX_FULLNAME` is correct.

#### 2. LLM Endpoint Access Issues
```bash
Error: Permission denied for endpoint
```
**Solution**: Verify endpoint permissions and correct endpoint name.

#### 3. Memory Issues During Optimization
```bash
Error: OutOfMemoryError
```
**Solution**: Reduce batch size, training set size, or use simpler optimizers.

#### 4. Slow Response Times
**Solution**: Use optimized configuration, reduce `top_k`, or enable caching.

### Debug Mode

Enable verbose logging for debugging:

```python
# In configuration
agent_config:
  verbose: true
  enable_tracing: true

# Or programmatically
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Features

### Custom Retrievers

```python
from dspy.retrieve.databricks_rm import DatabricksRM

# Custom retriever with specific settings
retriever = DatabricksRM(
    databricks_index_name="catalog.schema.index",
    text_column_name="content",
    docs_id_column_name="doc_id",
    columns=["doc_id", "title", "content"],
    k=10
)
```

### Multi-Stage RAG

```python
class MultiStageRAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.query_rewriter = dspy.ChainOfThought("query -> rewritten_query")
        self.response_generator = dspy.ChainOfThought("context, query -> response")
    
    def forward(self, query):
        # Rewrite query for better retrieval
        rewritten = self.query_rewriter(query=query)
        
        # Retrieve with rewritten query
        context = self.retriever(rewritten.rewritten_query)
        
        # Generate response
        response = self.response_generator(
            context=context.docs,
            query=query
        )
        
        return dspy.Prediction(response=response.response)
```

### Function Calling Integration

```python
from mlflow.models.resources import DatabricksFunction

# Add function resources to model
resources = [
    DatabricksServingEndpoint(endpoint_name="llm-endpoint"),
    DatabricksVectorSearchIndex(index_name="vector-index"),
    DatabricksFunction(function_name="catalog.schema.function_name")
]

mlflow.pyfunc.log_model(
    name="rag_agent_with_functions",
    python_model="agent.py",
    resources=resources,
    # ... other parameters
)
```

### Performance Monitoring

Currently working on incorporating evaluation pipeline using MLflow 3 evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Working with Databricks Notebooks

### Best Practices

1. **Run cells sequentially**: Each notebook is designed to be run from top to bottom
2. **Use widgets**: Many notebooks include Databricks widgets for easy configuration
3. **Monitor progress**: Watch for cell outputs and MLflow experiment tracking
4. **Restart when needed**: Use `dbutils.library.restartPython()` after installing packages

### Notebook Structure

Each notebook follows this pattern:
- **Setup cells**: Package installation (`%pip install`) and imports
- **Configuration cells**: Widget definitions and environment variables
- **Implementation cells**: Core logic and processing
- **Testing cells**: Validation and testing
- **Output cells**: Results and next steps

### Databricks-Specific Features Used

The notebooks leverage several Databricks-specific features:
- **Magic commands**: `%pip install`, `%md` for markdown cells
- **Widgets**: `dbutils.widgets` for interactive configuration
- **Display functions**: `display()` for rich table/chart outputs
- **Library restart**: `dbutils.library.restartPython()` after installations
- **Notebook context**: Access to user info and workspace details

### Tips for Success

- **Save frequently**: Use Databricks' auto-save feature
- **Use clusters appropriately**: Ensure your cluster has sufficient resources
- **Check dependencies**: Verify that previous notebooks have completed successfully
- **Monitor resources**: Watch memory and compute usage during optimization

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the Databricks documentation for Vector Search and Model Serving
- Refer to the DSPy documentation for advanced optimization techniques
- Consult Databricks notebook documentation for notebook-specific issues 