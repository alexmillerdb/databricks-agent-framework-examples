# Dual Virtual Environment Setup

This project uses two separate virtual environments with different Python versions:
- **MCP Environment**: Python 3.12 (for Model Context Protocol tools)
- **DSPy Environment**: Python 3.11 (for DSPy RAG Agent)

## Quick Start

### Initial Setup
```bash
# Make scripts executable
chmod +x activate-mcp.sh activate-dspy.sh env-switch.sh

# Initialize both environments
source env-switch.sh setup
```

### Switching Between Environments

#### Option 1: Direct Activation
```bash
# Activate MCP environment (Python 3.12)
source activate-mcp.sh

# Activate DSPy environment (Python 3.11)
source activate-dspy.sh
```

#### Option 2: Environment Switcher
```bash
# Show available environments and current status
source env-switch.sh status

# Switch to MCP environment
source env-switch.sh mcp

# Switch to DSPy environment
source env-switch.sh dspy
```

## Environment Details

### MCP Environment (Python 3.12)
- Location: `mcp/venv-3.12/`
- Requirements: `mcp/requirements.txt`
- Key packages: mlflow[databricks], mcp, databricks-mcp
- Main script: `mcp/databricks_mcp_example.py`

### DSPy Environment (Python 3.11)
- Location: `dspy/rag-agent/venv-3.11/`
- Requirements: `dspy/rag-agent/requirements.txt`
- Key packages: dspy-ai, mlflow[databricks], databricks-agents
- Main scripts: `dspy/rag-agent/agent.py`, `dspy/rag-agent/03-build-dspy-rag-agent.py`

## Key Improvements

1. **No Shell Spawning**: Scripts use `source` instead of `exec $SHELL`
2. **Auto-deactivation**: Automatically deactivates any active venv before switching
3. **Environment Tracking**: Sets `DATABRICKS_ACTIVE_ENV` variable
4. **Path Independence**: Scripts work from any directory
5. **Status Command**: Easy way to check which environment is active

## Troubleshooting

### Python Version Not Found
If you get "Python 3.11/3.12 not found" errors:
```bash
# macOS with Homebrew
brew install python@3.11 python@3.12

# Ubuntu/Debian
sudo apt-get install python3.11 python3.12
```

### Permission Denied
```bash
chmod +x activate-*.sh env-switch.sh
```

### Environment Variables
Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
# Edit .env with your Databricks credentials
```

## Best Practices

1. **Always use the correct environment** for each project:
   - MCP code → MCP environment
   - DSPy code → DSPy environment

2. **Check active environment** before installing packages:
   ```bash
   source env-switch.sh status
   ```

3. **Update dependencies** in the appropriate requirements.txt file

4. **Deactivate** when switching projects:
   ```bash
   deactivate
   ```
EOF < /dev/null