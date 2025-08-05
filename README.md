# databricks-agent-framework-examples
Databricks Agent Examples

This repository contains examples of building agents on Databricks using the Databricks Agent Framework, MLflow 3, and various AI frameworks.

## Examples

### DSPy RAG Agent (`dspy/rag-agent/`)

A comprehensive framework for building, optimizing, and deploying Retrieval-Augmented Generation (RAG) agents using DSPy and Databricks. This advanced implementation demonstrates state-of-the-art techniques for building production-ready RAG systems.

#### 🚀 Key Features

- **Advanced RAG Architecture**: Multi-stage pipeline with query rewriting, dynamic field mapping, and citation generation
- **DSPy Optimization**: Achieves **42.5% performance improvement** through sophisticated optimization techniques
- **Dedicated LLM Judge**: Separate Claude 3.7 Sonnet for optimization evaluation (critical for quality)
- **Modular Architecture**: Clean separation of concerns with dedicated modules for optimization, deployment, and utilities
- **Fast Optimization Mode**: New 5-10 minute optimization configuration for rapid development
- **Comprehensive Metrics**: Multi-dimensional evaluation framework measuring citation accuracy, semantic F1, completeness, and end-to-end performance
- **Production Ready**: Full MLflow integration with artifact management, model registry, and deployment to Model Serving endpoints

#### 📊 Performance Results

Through DSPy optimization with dedicated LLM judge, the framework achieves significant improvements:
- **Baseline Score**: 35.10%
- **Optimized Score**: 50.00%
- **Total Improvement**: +14.90 points **(42.5% improvement!)**

#### 🏗️ Recent Improvements

- **Modular Refactoring**: Main script reduced from 1,126 to 401 lines (65% reduction)
- **Enhanced Data Quality**: Comprehensive Wikipedia text cleaning (removes 22% noise)
- **Smart Resource Management**: Vector search endpoint/index checks and syncs
- **Deployment Safety**: Test logged models BEFORE deployment to endpoints
- **Fast Development Mode**: Bootstrap-only optimization for 5-10 minute iterations

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

#### 📁 Modular Architecture

The codebase follows a clean modular structure:

```
dspy/rag-agent/
├── 03-build-dspy-rag-agent.py    # Main orchestration (401 lines)
├── agent.py                       # MLflow ChatAgent
├── config/                        # Configuration files
├── modules/                       # Core functionality
│   ├── optimizer.py              # DSPy optimization
│   ├── deploy.py                 # MLflow deployment
│   └── utils.py                  # Utilities
└── tests/                         # Test suites
```

#### 🔧 Technical Implementation

- **Databricks Vector Search**: Efficient document retrieval at scale
- **MLflow ChatAgent**: Standardized agent interface for deployment
- **DSPy Framework**: Programmatic prompt optimization with dedicated judge LM
- **Multi-Stage Optimization**: Combines Bootstrap Few-Shot and MIPROv2 techniques
- **Comprehensive Testing**: Includes test suite with multiple test categories
- **Fast Mode**: 5-10 minute optimization for rapid iteration

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
