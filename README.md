# databricks-agent-framework-examples
Databricks Agent Examples

This repository contains examples of building agents on Databricks using the Databricks Agent Framework, MLflow 3, and various AI frameworks.

## Environment Management

This project uses **dual virtual environments** to support different Python versions:

- **DSPy Environment**: Python 3.11 (for DSPy RAG Agent examples)
- **MCP Environment**: Python 3.12 (for Model Context Protocol examples)

### Quick Setup

```bash
# Make scripts executable
chmod +x activate-mcp.sh activate-dspy.sh env-switch.sh

# Initialize both environments
source env-switch.sh setup
```

### Environment Switching

```bash
# Check current environment status
source env-switch.sh status

# Switch to MCP environment (Python 3.12)
source env-switch.sh mcp

# Switch to DSPy environment (Python 3.11)
source env-switch.sh dspy

# Direct activation (alternative)
source activate-mcp.sh    # MCP environment
source activate-dspy.sh   # DSPy environment
```

See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed setup instructions.

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

### MCP Examples (`mcp/`)

Examples for working with Databricks Model Context Protocol (MCP) using Python 3.12. Demonstrates:

- **OAuth Authentication**: Secure authentication using Databricks CLI profiles
- **Token Authentication**: Environment variable based authentication
- **Tool Discovery**: Lists and explores available MCP tools
- **Tool Execution**: Demonstrates calling MCP tools like `system__ai__python_exec`

#### Key Features

- **Python 3.12 Support**: Uses WorkspaceClient instead of databricks-connect
- **Multiple Auth Methods**: OAuth, token, and notebook authentication
- **Notebook Compatible**: Works as both script and Databricks notebook

See [mcp/README.md](mcp/README.md) for detailed documentation.

## Requirements

- **DSPy Examples**: Python 3.11, Databricks Runtime with MLflow 3.1+
- **MCP Examples**: Python 3.12, Databricks workspace with MCP enabled
- Unity Catalog enabled workspace
- Vector Search endpoint (for RAG examples)

## Installation

Each environment has its own requirements.txt file:

```bash
# For DSPy examples (Python 3.11)
source env-switch.sh dspy
pip install -r requirements.txt

# For MCP examples (Python 3.12)
source env-switch.sh mcp
pip install -r requirements.txt
```
