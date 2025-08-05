# Databricks MCP (Model Context Protocol) Examples

This directory contains examples of using the Databricks MCP client with Python 3.12.

## Files

- **`databricks_mcp_example.py`** - Example script/notebook demonstrating MCP usage
- **`mcp_agent.py`** - Production-ready MCP agent class
- **`deploy_mcp_agent.py`** - Deployment script for the MCP agent
- **`auth.py`** - Shared authentication utilities
- **`requirements.txt`** - Python dependencies for MCP functionality
- **`examples/`** - Specific examples for different MCP server types:
  - `vector_search_example.py` - Vector Search MCP example
  - `uc_functions_example.py` - Unity Catalog Functions example
  - `genie_space_example.py` - Genie Space example

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

### Local Development

1. Activate the MCP environment:
   ```bash
   source env-switch.sh mcp
   # or
   source activate-mcp.sh
   ```

2. Run the example:
   ```bash
   # With OAuth profile
   python databricks_mcp_example.py --profile myprofile
   
   # With token auth (requires .env)
   python databricks_mcp_example.py
   ```

### Databricks Notebook

1. Upload `databricks_mcp_example.py` to your Databricks workspace
2. Import as notebook (it contains magic commands and markdown cells)
3. Run all cells

The notebook includes:
- **Magic commands** for package installation (`%pip`)
- **Markdown documentation** cells (`%md`)
- **Automatic environment detection** (Databricks vs local)
- **Comprehensive error handling**
- **OAuth authentication** that works seamlessly in notebooks

### Server-Specific Examples

For specific MCP server types, use the examples in the `examples/` directory:

```bash
# Vector Search example
python examples/vector_search_example.py --catalog my_catalog --schema my_schema

# UC Functions example  
python examples/uc_functions_example.py --catalog my_catalog --schema my_schema

# Genie Space example
python examples/genie_space_example.py --genie-space-id my_genie_space_id

# All examples support --profile for OAuth
python examples/vector_search_example.py --profile myprofile --catalog my_catalog --schema my_schema
```

## Features

### Authentication Methods
- **OAuth via CLI profiles** - Secure authentication using Databricks CLI
- **Token authentication** - Environment variable based auth for automation
- **Notebook authentication** - Automatic auth in Databricks notebooks

### MCP Client Functionality
- **Tool Discovery** - Lists all available MCP tools across different server types
- **Tool Execution** - Execute MCP tools with proper parameter handling
- **Multiple Server Types** - Support for System AI, Vector Search, UC Functions, and Genie
- **Production Agent** - `mcp_agent.py` provides a complete agent class
- **Deployment Ready** - `deploy_mcp_agent.py` deploys agents to Model Serving
- **Multiple auth methods** - Works with profiles, tokens, or default auth

## Example Output

```
🚀 Databricks MCP Example
==================================================
🔌 Testing MCP Server Connection...
==================================================
☁️  Databricks Environment Mode
========================================
ℹ️  Using Databricks workspace credentials

👤 User: your.email@company.com
🌐 Connecting to MCP server: https://your-workspace.cloud.databricks.com/api/2.0/mcp/functions/system/ai

🔍 Discovering available tools...
✅ Discovered 3 tools: ['system__ai__python_exec', 'system__ai__sql_exec', 'system__ai__file_search']

🧪 Testing system__ai__python_exec tool...
✅ Tool execution result:
Hello from MCP! Current time: 2024-08-05 13:45:23.123456
Python version: 3.12.0 (main, Oct  2 2023, 13:45:54) [GCC 11.4.0] on linux
MCP tool execution successful! 🎉
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```
   ❌ Failed to create WorkspaceClient: Invalid authentication
   ```
   **Solution**: Check your Databricks CLI configuration or environment variables

2. **MCP Server Not Available**
   ```
   ❌ Error testing MCP server: Connection refused
   ```
   **Solution**: Ensure MCP is enabled in your workspace and you have proper permissions

3. **Missing Dependencies**
   ```
   ImportError: No module named 'databricks_mcp'
   ```
   **Solution**: Install dependencies with `uv pip install -r requirements.txt`

### Environment Issues

If you see Python version errors:
```bash
# Check available Python versions
ls /usr/bin/python*

# Install Python 3.12 (macOS)
brew install python@3.12

# Install Python 3.12 (Ubuntu)
sudo apt-get install python3.12
```

### Debug Mode

Add debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Production MCP Agent

Use the `DatabricksMCPAgent` class for production workflows:

```python
from mcp_agent import create_agent

# Create agent for different server types
system_agent = create_agent(profile="myprofile", server_type="system/ai")
vector_agent = create_agent(
    profile="myprofile", 
    server_type="vector-search", 
    catalog="my_catalog", 
    schema="my_schema"
)

# Execute tools
result = system_agent.execute_python_code("print('Hello from MCP!')")
tools = vector_agent.list_available_tools()
```

### Agent Deployment

Deploy the MCP agent to Databricks Model Serving:

```bash
# Deploy with OAuth profile
python deploy_mcp_agent.py --profile myprofile

# Deploy without serving endpoint (log and register only)
python deploy_mcp_agent.py --profile myprofile --skip-deploy
```

### Custom Tool Calls

```python
# Example: System AI function
result = mcp_client.call_tool(
    "system__ai__sql_exec", 
    {"query": "SELECT COUNT(*) FROM your_table"}
)

# Example: Vector Search
result = mcp_client.call_tool(
    "search_vectors", 
    {"query": "machine learning", "top_k": 5}
)

# Example: UC Function
result = mcp_client.call_tool(
    "my_custom_function", 
    {"param1": "value1", "param2": "value2"}
)
```

### Exploring Available Tools

```python
tools = mcp_client.list_tools()
for tool in tools:
    print(f"Tool: {tool.name}")
    if hasattr(tool, 'description'):
        print(f"Description: {tool.description}")
    if hasattr(tool, 'inputSchema'):
        print(f"Schema: {tool.inputSchema}")
```

## Next Steps

1. **Explore Server Types** - Try different MCP server types (vector-search, functions, genie)
2. **Deploy Production Agents** - Use `deploy_mcp_agent.py` to deploy to Model Serving
3. **Build Custom UC Functions** - Create your own functions and access via MCP
4. **Integrate with DSPy** - Combine MCP tools with DSPy RAG agents
5. **Create Custom Workflows** - Build complex workflows using multiple MCP server types

## Architecture Overview

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   MCP Examples      │    │   Shared Auth        │    │   Databricks        │
│                     │    │   (auth.py)          │    │   Workspace         │
├─────────────────────┤    ├──────────────────────┤    ├─────────────────────┤
│ • databricks_mcp_   │───▶│ • OAuth/CLI profiles │───▶│ • System AI         │
│   example.py        │    │ • Token auth         │    │ • Vector Search     │
│ • mcp_agent.py      │    │ • Notebook auth      │    │ • UC Functions      │
│ • deploy_mcp_       │    │ • URL builders       │    │ • Genie Spaces      │
│   agent.py          │    │ • Client creation    │    │                     │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

For more information, see the [Databricks MCP documentation](https://docs.databricks.com/en/dev-tools/mcp.html).