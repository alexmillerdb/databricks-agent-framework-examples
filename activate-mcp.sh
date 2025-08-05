#!/bin/bash
# Activation script for MCP environment (Python 3.12)

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MCP_DIR="$SCRIPT_DIR/mcp"

echo "🚀 Activating MCP Environment..."
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "$MCP_DIR/venv-3.12" ]; then
    echo "❌ Virtual environment not found at $MCP_DIR/venv-3.12"
    echo "   Please run: cd mcp && python3.12 -m venv venv-3.12"
    return 1 2>/dev/null || exit 1
fi

# Deactivate any active virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "⚠️  Deactivating current environment: $VIRTUAL_ENV"
    deactivate
fi

# Activate the virtual environment
source "$MCP_DIR/venv-3.12/bin/activate"

# Set environment variable to indicate which env is active
export DATABRICKS_ACTIVE_ENV="mcp"

echo "✅ Activated MCP environment"
echo "🐍 Python version: $(python --version)"
echo "📍 Working directory: $MCP_DIR"
echo "🎯 Virtual env: $VIRTUAL_ENV"
echo ""
echo "💡 To deactivate, run: deactivate"
echo "📚 Available commands:"
echo "  - python databricks_mcp_example.py"
echo "  - pip install -r requirements.txt"
echo ""

# Change to MCP directory
cd "$MCP_DIR"