# databricks-agent-framework-examples
Databricks Agent Examples

This repository contains examples of building agents on Databricks using the Databricks Agent Framework, MLflow 3, and various AI frameworks.

## Examples

### DSPy RAG Agent (`dspy/rag-agent/`)

A comprehensive framework for building, optimizing, and deploying Retrieval-Augmented Generation (RAG) agents using DSPy and Databricks. Features include:

- **DSPy Integration**: Build and optimize language model programs
- **Databricks Vector Search**: Efficient document retrieval at scale
- **MLflow ChatAgent**: Standardized agent interface for deployment
- **Production Ready**: Full pipeline from development to deployment

See [dspy/rag-agent/README.md](dspy/rag-agent/README.md) for detailed documentation.

## Requirements

- Python 3.10+
- Databricks Runtime with MLflow 3.1+
- Unity Catalog enabled workspace
- Vector Search endpoint (for RAG examples)

## Installation

```bash
pip install -r requirements.txt
```

**Note**: Requirements pin Pydantic to `>=2.5.0,<2.12` to avoid compatibility issues with MLflow validation.
