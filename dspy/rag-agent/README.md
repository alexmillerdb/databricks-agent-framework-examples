# DSPy RAG Agent Framework

A comprehensive framework for building, optimizing, and deploying Retrieval-Augmented Generation (RAG) agents using DSPy and Databricks. This framework provides a complete end-to-end pipeline from data preparation to production deployment.

## 🎯 Key Achievements

- **172% Performance Improvement**: From 18.33% to 49.80% accuracy through DSPy optimization
- **Advanced RAG Architecture**: Query rewriting, dynamic field mapping, and citation generation
- **Production Ready**: Full MLflow integration with deployment to Model Serving endpoints
- **Comprehensive Metrics**: Multi-dimensional evaluation framework for RAG quality
- **Rapid Development**: Includes test suite and modular design for quick iteration

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
│   Data Prep     │───▶│   Vector Search │───▶│  Unified Build  │
│   (Step 1)      │    │   Index (Step 1)│    │ & Optimization  │
└─────────────────┘    └─────────────────┘    │   (Step 2)      │
                                               └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   Deployment    │
                                               │   (Step 2)      │
                                               └─────────────────┘
```

### Enhanced RAG Architecture

The framework implements an advanced multi-stage RAG architecture with query rewriting and optimized retrieval:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  User Query     │───▶│ Query Rewriter  │───▶│ Vector Search   │
│                 │    │  (Optimized)    │    │   Retrieval     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │ Response Gen.   │
                                               │ (w/ Citations)  │
                                               └─────────────────┘
```

#### Key Components:
1. **Query Rewriter**: Transforms user queries into optimized search queries
2. **Dynamic Field Mapping**: Configurable field names for different vector stores
3. **Citation Generator**: Adds proper citations [1], [2] to responses
4. **Multi-Metric Evaluation**: Comprehensive evaluation framework

### Key Components

- **agent.py**: MLflow ChatAgent with optimized program loading via `load_context()`
- **utils.py**: Retriever building and robust optimized program loading
- **metrics.py**: Comprehensive evaluation metrics for optimization
- **03-build-dspy-rag-agent.py**: Unified build, optimization, and deployment workflow
- **config.yaml**: Single configuration file for all settings

## Current File Structure

```
dspy/rag-agent/
├── 01-dspy-data-preparation.py     # Data prep & vector search index
├── 02-create-eval-dataset.py      # Optional: create evaluation dataset
├── 03-build-dspy-rag-agent.py     # Main workflow: build, optimize, deploy
├── agent.py                       # MLflow ChatAgent implementation
├── utils.py                       # Helper functions for retrieval & loading
├── metrics.py                     # Comprehensive evaluation metrics
├── test_agent.py                  # Test suite for rapid development
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── optimized_rag_program.json     # Generated: optimized DSPy program
├── program_metadata.json          # Generated: optimization metadata
└── README.md                      # This documentation
```

## Prerequisites

### Environment Requirements

- **Databricks Runtime**: Serverless CPU with version 2 (old version used 3, ensure you are using 2)
- **Databricks Workspace**: With Unity Catalog enabled
- **Vector Search Endpoint**: Pre-configured Databricks Vector Search endpoint

### Python Dependencies

Install in Databricks notebook cells:

```python
%pip install --upgrade "mlflow[databricks]>=3.1" dspy-ai>=2.5 databricks-agents>=0.5 openai uv
dbutils.library.restartPython()
```

**Important**: The requirements.txt file pins Pydantic to `>=2.5.0,<2.12` to avoid compatibility issues with pre-release versions that can cause validation errors in MLflow.

### Required Permissions

- Unity Catalog: `CREATE`, `READ`, `WRITE` permissions on target catalog/schema
- Vector Search: Access to create and query vector search indexes
- Model Serving: Permissions to deploy models to serving endpoints

## Quick Start

### 1. Import Notebooks to Databricks

1. Upload the notebook files to your Databricks workspace
2. Or clone the repository and import the `dspy/rag-agent/` folder

### 2. Configure Environment Variables (if running locally in IDE)

Set these variables in your IDE .env file

```
# Databricks Personal Access Token
# Generate this from your Databricks workspace: User Settings > Developer > Access tokens
DATABRICKS_TOKEN=databricks token

# Databricks Host URL
# Your workspace URL (e.g., https:/<>.databricks.com)
DATABRICKS_HOST=databricks host

# MLflow Tracking Configuration
MLFLOW_TRACKING_URI=databricks

# MLflow Experiment ID
# Create an experiment in your Databricks workspace and use its ID here
MLFLOW_EXPERIMENT_ID=experiment ID in Databricks

# Databricks Serverless Compute is enabled by default
DATABRICKS_SERVERLESS_COMPUTE_ID=auto
```

### 3. Run the Complete Pipeline

Open and run the notebooks in order in your Databricks workspace:

1. **Step 1**: `01-dspy-data-preparation.py` - Data preparation and vector search index creation
2. **Step 2**: `03-build-dspy-rag-agent.py` - Unified workflow: build, optimize, and deploy agent

**Optional**: 
- `02-create-eval-dataset.py` - Create custom evaluation dataset for optimization

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

### Step 2: Unified Build, Optimization, and Deployment

**Notebook**: `03-build-dspy-rag-agent.py`

This comprehensive notebook combines agent creation, optimization, and deployment into a single streamlined workflow.

#### What it does:
- **Environment Setup**: Configures for local or Databricks execution
- **Base Agent Creation**: Builds DSPy RAG agent with MLflow ChatAgent interface
- **Optimization (Optional)**: Uses DSPy compilation to improve performance
- **MLflow Logging**: Registers the agent with proper artifact handling
- **Deployment (Optional)**: Deploys to Model Serving endpoint

#### Key Features:
- **Unified Workflow**: Everything in one notebook for simplicity
- **MLflow Best Practices**: Proper artifact handling for optimized programs
- **Robust Loading**: Uses `load_context()` for reliable deployment
- **Configuration Driven**: Single `config.yaml` controls all settings
- **Production Ready**: Includes validation and deployment

#### Configuration Options:
```python
# Workflow Control Parameters
OPTIMIZE_AGENT = True          # Whether to run DSPy optimization
DEPLOY_MODEL = True            # Whether to deploy to Model Serving
EVAL_DATASET_NAME = "wikipedia_synthetic_eval"  # Name of evaluation dataset

# Unity Catalog Configuration  
UC_CATALOG = "users"
UC_SCHEMA = "alex_miller"
UC_MODEL_NAME = "dspy_rag_agent"
```

#### Running the notebook:
1. Open `03-build-dspy-rag-agent.py` in your Databricks workspace
2. Configure the parameters in the configuration section
3. Run all cells sequentially - the notebook will:
   - Build the base RAG agent
   - Optionally optimize using DSPy compilation
   - Log the model with proper artifact handling
   - Validate the model for deployment
   - Optionally deploy to Model Serving

#### Expected output:
- Base RAG agent with retrieval and generation capabilities
- Optimized program (if enabled) with performance improvements
- MLflow model registered in Unity Catalog
- Deployed Model Serving endpoint (if enabled)
- Performance metrics and optimization results

### Optional: Create Custom Evaluation Dataset

**Notebook**: `02-create-eval-dataset.py`

This optional notebook helps you create custom evaluation datasets for better optimization results.

#### What it does:
- Creates synthetic question-answer pairs from your data
- Generates evaluation criteria and expected responses
- Stores the dataset in Unity Catalog for use in optimization

#### When to use:
- You want to optimize for domain-specific performance
- You have specific quality criteria for your RAG responses
- You want more control over the optimization process

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

# DSPy Configuration
dspy_config:
  response_generator_signature: "context, request -> response"
  optimized_program_path: "optimized_rag_program.json"  # Fallback path

# Agent Configuration
agent_config:
  use_optimized: true
  use_query_rewriter: true  # Enable query rewriting for better retrieval
  enable_tracing: true
  verbose: true
  max_iterations: 10

# MLflow Configuration
mlflow_config:
  enable_autolog: true
  experiment_name: "dspy-rag-agent"
  registered_model_name: "dspy_rag_chat_agent"
```

### Optimized Program Loading

The framework uses MLflow's best practices for loading optimized DSPy programs:

#### In Development:
- Programs are loaded from local files during optimization
- Paths stored in `agent_config.optimized_program_path`

#### In Deployment:
- Programs are loaded via MLflow artifacts using `load_context()`
- Automatic path resolution in deployment environment
- Robust fallback chain: MLflow artifacts → config paths → environment variables

#### Configuration for Optimized Loading:
```yaml
agent_config:
  use_optimized: true
  optimized_program_artifact: "optimized_program"  # MLflow artifact key
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

### 🚀 Optimization Results

The framework achieves **remarkable performance improvements** through DSPy optimization:

- **Baseline Score**: 18.33%
- **Optimized Score**: 49.80%
- **Total Improvement**: +31.47 points **(172% improvement!)**

This dramatic improvement demonstrates the power of DSPy's optimization techniques combined with our multi-stage RAG architecture.

### DSPy Optimization Techniques

The framework implements a **multi-stage optimization strategy** that combines multiple approaches:

#### 1. Multi-Stage Optimization Pipeline
```python
OPTIMIZATION_CONFIG = {
    "strategy": "multi_stage",      # Combines multiple optimizers
    "auto_level": "light",          # "light", "medium", or "heavy"
    "num_threads": 2,               # Concurrent optimization threads
    "training_examples_limit": 10,  # Training set size
    "evaluation_examples_limit": 5, # Evaluation set size
}
```

#### 2. Bootstrap Few-Shot Optimization
- **Few-shot learning**: Learns from provided examples
- **Bootstrapping**: Generates additional training examples
- **Demonstrated improvement**: +26.67 points in our tests

```python
optimizer = dspy.BootstrapFewShot(
    metric=rag_evaluation_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=2,
    metric_threshold=0.6
)
```

#### 3. MIPROv2 (Advanced)
- **Multi-stage optimization**: Iteratively improves instructions and demonstrations
- **Automatic mode selection**: Chooses appropriate optimization strategy
- **Performance tracking**: Monitors optimization progress

```python
optimizer = dspy.MIPROv2(
    metric=rag_evaluation_metric,
    auto="light",  # Cannot set num_candidates when auto is specified
    init_temperature=1.0,
    num_threads=2,
    verbose=True,
    track_stats=True
)
```

### Comprehensive Evaluation Metrics

The framework includes a sophisticated metrics system (`metrics.py`) that evaluates multiple aspects:

#### 1. Citation Accuracy Metric
- Validates proper citation format [1], [2], etc.
- Ensures citations are sequential and valid
- Checks citation consistency

#### 2. Semantic F1 Metric
- Extracts facts from responses
- Compares predicted vs expected facts
- Supports fuzzy matching for flexibility

#### 3. Completeness Metric
- Measures if all question keywords are addressed
- Configurable threshold (default: 60%)
- Filters out common question words

#### 4. End-to-End RAG Metric
- Combines multiple metrics with weights:
  - Citation accuracy: 20%
  - Retrieval relevance: 30%
  - Semantic F1: 40%
  - Completeness: 10%

### Custom Evaluation Implementation

```python
def rag_evaluation_metric(example, prediction, trace=None):
    """Advanced RAG evaluation using LLM-as-judge."""
    
    # Comprehensive evaluation using DSPy signatures
    eval_result = evaluate_response(
        request=example.request,
        response=prediction.response,
        expected_facts=example.response
    )
    
    # Extract scores
    factual_accuracy = float(eval_result.factual_accuracy)
    completeness = float(eval_result.completeness)
    overall_score = float(eval_result.overall_score)
    
    return overall_score  # Returns 0.0-1.0
```

### Query Rewriter Optimization

The query rewriter significantly improves retrieval quality:

**Before**: "Who started heavy metal music?"
**After**: "Who is the founder of heavy metal music in the 1960s or 1970s?"

**Before**: "What is Greek mythology?"
**After**: "What is the definition of ancient Greek mythology, including its stories, legends, and mythological tales?"

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

#### 5. Pydantic Validation Errors
```bash
AttributeError: 'pydantic_core._pydantic_core.ValidationInfo' object has no attribute 'content'
```
**Solution**: This occurs when MLflow's validation environment installs incompatible Pydantic versions (e.g., 2.12.0a1). The framework includes fixes:
- Requirements pin Pydantic to `>=2.5.0,<2.12`
- Agent code handles both dict and ChatAgentMessage inputs
- Model validation warnings are non-fatal

#### 6. Optimized Program Not Loading
```bash
No optimized program found, using default
```
**Solution**: This indicates the optimized program isn't loading correctly. Check:
- MLflow artifacts are properly logged with key "optimized_program"
- `agent_config.use_optimized` is set to `true`
- The optimized program file exists and is valid JSON
- Check MLflow logs for detailed loading attempts

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

### MLflow Artifact Integration

The framework demonstrates MLflow best practices for handling optimized programs:

```python
class DSPyRAGChatAgent(ChatAgent):
    def load_context(self, context):
        """MLflow calls this during model loading."""
        if "optimized_program" in context.artifacts:
            artifact_path = context.artifacts["optimized_program"]
            optimized_program = self._load_optimized_from_artifact(artifact_path)
            if optimized_program:
                self.rag = optimized_program
```

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

### Multi-Stage RAG Implementation

The framework implements a sophisticated multi-stage RAG architecture:

```python
class _DSPyRAGProgram(dspy.Module):
    def __init__(self, retriever, config=None):
        super().__init__()
        self.retriever = retriever
        self.config = config or {}
        
        # Dynamic field mapping from config
        vs_config = self.config.get("vector_search", {})
        self.text_field = vs_config.get("text_column_name", "chunk")
        self.id_field = vs_config.get("docs_id_column_name", "id")
        self.columns = vs_config.get("columns", ["id", "title", "chunk_id"])
        
        # Query rewriter for better retrieval
        if self.use_query_rewriter:
            self.query_rewriter = dspy.ChainOfThought(RewriteQuery)
            
        # Response generator with citations
        self.response_generator = dspy.ChainOfThought(GenerateCitedAnswer)
    
    def forward(self, request: str):
        # Step 1: Rewrite query if enabled
        if self.use_query_rewriter:
            rewritten = self.query_rewriter(original_question=request)
            search_query = rewritten.rewritten_query
        else:
            search_query = request
            
        # Step 2: Retrieve with optimized query
        retrieved_context = self.retriever(search_query)
        
        # Step 3: Generate response with citations
        response = self.response_generator(
            context=retrieved_context.docs,
            question=request
        )
        
        return dspy.Prediction(response=response.answer)
```

### Key Improvements in This Implementation

1. **Query Rewriting**: The `RewriteQuery` signature optimizes user queries for better retrieval
2. **Dynamic Field Mapping**: Automatically adapts to different vector store schemas
3. **Citation Generation**: The `GenerateCitedAnswer` signature ensures proper citation format
4. **Configuration-Driven**: All components configurable via `config.yaml`
5. **Optimization-Ready**: Structure designed for DSPy compilation

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