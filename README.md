# databricks-agent-framework-examples
Databricks Agent Examples

This repository contains examples of building agents on Databricks using the Databricks Agent Framework, MLflow 3, and various AI frameworks.

## Examples

### DSPy RAG Agent (`dspy/rag-agent/`)

A comprehensive framework for building, optimizing, and deploying Retrieval-Augmented Generation (RAG) agents using DSPy and Databricks. This advanced implementation demonstrates state-of-the-art techniques for building production-ready RAG systems.

#### 🚀 Key Features

- **Advanced RAG Architecture**: Multi-stage pipeline with query rewriting, dynamic field mapping, and citation generation
- **DSPy Optimization**: Achieves **172% performance improvement** through sophisticated optimization techniques
- **Comprehensive Metrics**: Multi-dimensional evaluation framework measuring citation accuracy, semantic F1, completeness, and end-to-end performance
- **Production Ready**: Full MLflow integration with artifact management, model registry, and deployment to Model Serving endpoints

#### 📊 Performance Results

Through DSPy optimization, the framework achieves remarkable improvements:
- **Baseline Score**: 18.33%
- **Optimized Score**: 49.80%
- **Total Improvement**: +31.47 points **(172% improvement!)**

#### 🏗️ Architecture Components

1. **Query Rewriter Module**: Transforms user queries into optimized search queries
   - Example: "Who started heavy metal?" → "Who is the founder of heavy metal music in the 1960s or 1970s?"
   
2. **Dynamic Field Mapping**: Configurable field names for different vector stores
   - Supports custom schemas (chunk, content, text, etc.)
   - Automatic metadata field extraction
   
3. **Citation Generator**: Adds proper citations [1], [2] to responses
   - Validates citation format and consistency
   - Ensures sequential numbering
   
4. **Multi-Metric Evaluation System**:
   - **Citation Accuracy**: Validates proper citation usage
   - **Semantic F1**: Measures fact coverage and accuracy
   - **Completeness**: Ensures all query aspects are addressed
   - **End-to-End Score**: Weighted combination of all metrics

#### 🔧 Technical Implementation

- **Databricks Vector Search**: Efficient document retrieval at scale
- **MLflow ChatAgent**: Standardized agent interface for deployment
- **DSPy Framework**: Programmatic prompt optimization
- **Multi-Stage Optimization**: Combines Bootstrap Few-Shot and MIPROv2 techniques
- **Comprehensive Testing**: Includes test suite with 13 test categories

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
