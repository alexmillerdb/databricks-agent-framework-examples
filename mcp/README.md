# Databricks MCP (Model Context Protocol) Agent

This directory contains a production-ready MCP ResponsesAgent implementation for deployment on Databricks Model Serving, following the official Databricks MCP documentation.

## Core Files

- **`mcp_responses_agent.py`** - Production ResponsesAgent implementation for deployment
- **`deploy_responses_agent.py`** - Deployment script for Databricks Model Serving
- **`mcp_agent/`** - Authentication utilities package
- **`requirements.txt`** - Python dependencies for MCP functionality
- **`databricks-mcp-documentation.md`** - Official Databricks MCP documentation reference

## Prerequisites

1. **Python 3.12 or higher** (databricks-connect does not support 3.12 yet)
2. **Databricks workspace** with Unity Catalog enabled
3. **Appropriate permissions** to access Unity Catalog functions
4. **MCP server** configured in your workspace

## Environment Setup

This MCP example requires Python 3.12 and its own virtual environment, separate from the DSPy examples.

### Quick Setup

From the project root directory:

```bash
# Initialize MCP environment
source env-switch.sh setup

# Activate MCP environment
source env-switch.sh mcp

# Verify Python version
python --version  # Should show Python 3.12.x
```

### Manual Setup

If you prefer manual setup:

```bash
cd mcp
python3.12 -m venv venv-3.12
source venv-3.12/bin/activate
pip install -r requirements.txt
```

## Shared Authentication

All MCP examples use a shared authentication module (`auth.py`) that provides:

- **`setup_workspace_client()`** - Configure WorkspaceClient with multiple auth methods
- **`get_current_user()`** - Get current user from workspace
- **`get_mcp_server_url()`** - Build MCP server URLs for different server types
- **`setup_mlflow_for_profile()`** - Configure MLflow for profiles
- **`create_mcp_client_with_auth()`** - Create authenticated MCP clients

### MCP Server Types Supported

The authentication utilities support all Databricks MCP server types:

1. **System AI Functions** (`system/ai`):
   - URL: `https://<workspace>/api/2.0/mcp/functions/system/ai` 
   - Built-in AI functions like python_exec, sql_exec, file_search

2. **Vector Search** (`vector-search`):
   - URL: `https://<workspace>/api/2.0/mcp/vector-search/{catalog}/{schema}`
   - Access to Vector Search indexes in Unity Catalog

3. **Unity Catalog Functions** (`functions`):
   - URL: `https://<workspace>/api/2.0/mcp/functions/{catalog}/{schema}`
   - Custom UC functions in your catalog/schema

4. **Genie Space** (`genie`):
   - URL: `https://<workspace>/api/2.0/mcp/genie/{genie_space_id}`
   - Genie AI assistant for specific spaces

## Authentication Setup

Since `databricks-connect` doesn't support Python 3.12 yet, this example uses OAuth/WorkspaceClient authentication.

### Option 1: OAuth Authentication (Recommended)

1. Install and configure Databricks CLI:
   ```bash
   pip install databricks-cli
   databricks auth login --profile myprofile
   ```

2. Run the example with profile:
   ```bash
   python databricks_mcp_example.py --profile myprofile
   ```

### Option 2: Token Authentication

1. Create a `.env` file:
   ```env
   DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
   DATABRICKS_TOKEN=your-databricks-token
   ```

2. Run the example:
   ```bash
   python databricks_mcp_example.py
   ```

### Option 3: Databricks Notebook

1. Upload `databricks_mcp_example.py` to your Databricks workspace
2. Run the notebook - it will use the notebook's authentication context automatically

## Usage

### Deploy MCP Agent

Deploy the MCP ResponsesAgent to Databricks Model Serving:

```bash
# Deploy with OAuth profile
python deploy_responses_agent.py --profile myprofile

# Deploy to specific catalog/schema
python deploy_responses_agent.py --profile myprofile --catalog main --schema default

# Log and register only (skip deployment)
python deploy_responses_agent.py --profile myprofile --skip-deploy
```

### Test Local Agent

Test the ResponsesAgent locally before deployment:

```bash
python mcp_responses_agent.py
```

### Use Deployed Model

Once deployed, use the model via MLflow:

```python
import mlflow

# Load the deployed model
model = mlflow.pyfunc.load_model("models:/catalog.schema.mcp_responses_agent/1")

# Make predictions
response = model.predict({
    "input": [{"role": "user", "content": "What's the 100th Fibonacci number?"}]
})
```

## Features

### Authentication Methods
- **OAuth via CLI profiles** - Secure authentication using Databricks CLI
- **Token authentication** - Environment variable based auth for automation
- **Notebook authentication** - Automatic auth in Databricks notebooks

### MCP ResponsesAgent Features
- **Tool Discovery** - Automatically discovers tools from configured MCP servers
- **Tool Execution** - Single-turn agent pattern with LLM + tool calling
- **Multiple Server Types** - Support for System AI, Vector Search, UC Functions, and Genie
- **Production Ready** - Full ResponsesAgent implementation following Databricks documentation
- **Auto-deployment** - Automatic resource discovery and Model Serving deployment
- **Authentication** - OAuth, token, and notebook authentication support

## Configuration

### Environment Variables

You can configure the MCP agent using environment variables:

```bash
# LLM endpoint for agent reasoning
export LLM_ENDPOINT_NAME="databricks-claude-3-7-sonnet"

# System prompt for the agent
export SYSTEM_PROMPT="You are a helpful assistant with access to various tools."

# Databricks CLI profile (optional)
export DATABRICKS_CLI_PROFILE="myprofile"

# Additional MCP server URLs (comma-separated)
export MCP_SERVER_URLS="https://workspace/api/2.0/mcp/vector-search/catalog/schema"
```

### Supported MCP Server Types

The agent automatically configures URLs for different MCP server types:

| Server Type | URL Pattern | Description |
|-------------|-------------|-------------|
| System AI | `/api/2.0/mcp/functions/system/ai` | Built-in AI functions (python_exec, sql_exec) |
| Vector Search | `/api/2.0/mcp/vector-search/{catalog}/{schema}` | Search vector indexes |
| UC Functions | `/api/2.0/mcp/functions/{catalog}/{schema}` | Custom Unity Catalog functions |
| Genie Space | `/api/2.0/mcp/genie/{space_id}` | Genie AI assistant |

## Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   ResponsesAgent    │    │   MCP Servers        │    │   Databricks        │
│                     │    │                      │    │   Model Serving     │
├─────────────────────┤    ├──────────────────────┤    ├─────────────────────┤
│ • mcp_responses_    │───▶│ • System AI          │───▶│ • Auto-scaling      │
│   agent.py          │    │ • Vector Search      │    │ • Authentication    │
│ • Tool discovery    │    │ • UC Functions       │    │ • Resource mgmt     │
│ • LLM integration   │    │ • Genie Spaces       │    │ • Monitoring        │
│ • Single-turn flow  │    │ • Auto-discovery     │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

For more information, see the [Databricks MCP documentation](https://docs.databricks.com/en/dev-tools/mcp.html).