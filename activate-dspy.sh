#!/bin/bash
# Activation script for DSPy RAG Agent environment (Python 3.11)

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DSPY_DIR="$SCRIPT_DIR/dspy/rag-agent"

echo "🚀 Activating DSPy RAG Agent Environment..."
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "$DSPY_DIR/venv-3.11" ]; then
    echo "❌ Virtual environment not found at $DSPY_DIR/venv-3.11"
    echo "   Please run: cd dspy/rag-agent && python3.11 -m venv venv-3.11"
    return 1 2>/dev/null || exit 1
fi

# Deactivate any active virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "⚠️  Deactivating current environment: $VIRTUAL_ENV"
    deactivate
fi

# Activate the virtual environment
source "$DSPY_DIR/venv-3.11/bin/activate"

# Set environment variable to indicate which env is active
export DATABRICKS_ACTIVE_ENV="dspy"

echo "✅ Activated DSPy RAG Agent environment"
echo "🐍 Python version: $(python --version)"
echo "📍 Working directory: $DSPY_DIR"
echo "🎯 Virtual env: $VIRTUAL_ENV"
echo ""
echo "💡 To deactivate, run: deactivate"
echo "📚 Available commands:"
echo "  - python agent.py"
echo "  - python 03-build-dspy-rag-agent.py"
echo "  - pip install -r requirements.txt"
echo ""

# Change to DSPy directory
cd "$DSPY_DIR"