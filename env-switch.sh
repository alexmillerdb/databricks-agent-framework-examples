#!/bin/bash
# Master environment switcher for Databricks Agent Framework Examples

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_usage() {
    echo "🔄 Databricks Agent Framework Environment Switcher"
    echo "================================================="
    echo ""
    echo "Usage: source env-switch.sh [mcp|dspy|status|setup]"
    echo ""
    echo "Commands:"
    echo "  mcp     - Activate MCP environment (Python 3.12)"
    echo "  dspy    - Activate DSPy environment (Python 3.11)"
    echo "  status  - Show current environment status"
    echo "  setup   - Initialize both virtual environments"
    echo ""
    echo "Current Status:"
    show_status
}

show_status() {
    echo ""
    if [ -n "$VIRTUAL_ENV" ]; then
        echo -e "${GREEN}✅ Active Environment:${NC} $DATABRICKS_ACTIVE_ENV"
        echo -e "${GREEN}   Virtual Env Path:${NC} $VIRTUAL_ENV"
        echo -e "${GREEN}   Python Version:${NC} $(python --version 2>&1)"
    else
        echo -e "${YELLOW}⚠️  No virtual environment active${NC}"
    fi
    
    echo ""
    echo "Available Environments:"
    
    # Check MCP environment
    if [ -d "$SCRIPT_DIR/mcp/venv-3.12" ]; then
        echo -e "${GREEN}  ✓ MCP${NC} (Python 3.12) - $SCRIPT_DIR/mcp/venv-3.12"
    else
        echo -e "${RED}  ✗ MCP${NC} (Python 3.12) - Not initialized"
    fi
    
    # Check DSPy environment
    if [ -d "$SCRIPT_DIR/dspy/rag-agent/venv-3.11" ]; then
        echo -e "${GREEN}  ✓ DSPy${NC} (Python 3.11) - $SCRIPT_DIR/dspy/rag-agent/venv-3.11"
    else
        echo -e "${RED}  ✗ DSPy${NC} (Python 3.11) - Not initialized"
    fi
}

setup_environments() {
    echo "🔧 Setting up virtual environments..."
    echo "===================================="
    
    # Setup MCP environment
    echo ""
    echo "📦 Setting up MCP environment (Python 3.12)..."
    if command -v python3.12 &> /dev/null; then
        cd "$SCRIPT_DIR/mcp"
        python3.12 -m venv venv-3.12
        source venv-3.12/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        deactivate
        echo -e "${GREEN}✅ MCP environment created successfully${NC}"
    else
        echo -e "${RED}❌ Python 3.12 not found. Please install it first.${NC}"
    fi
    
    # Setup DSPy environment
    echo ""
    echo "📦 Setting up DSPy environment (Python 3.11)..."
    if command -v python3.11 &> /dev/null; then
        cd "$SCRIPT_DIR/dspy/rag-agent"
        python3.11 -m venv venv-3.11
        source venv-3.11/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        deactivate
        echo -e "${GREEN}✅ DSPy environment created successfully${NC}"
    else
        echo -e "${RED}❌ Python 3.11 not found. Please install it first.${NC}"
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
    echo "🎉 Setup complete!"
    show_status
}

# Main logic
case "$1" in
    mcp)
        source "$SCRIPT_DIR/activate-mcp.sh"
        ;;
    dspy)
        source "$SCRIPT_DIR/activate-dspy.sh"
        ;;
    status)
        show_status
        ;;
    setup)
        setup_environments
        ;;
    *)
        show_usage
        ;;
esac