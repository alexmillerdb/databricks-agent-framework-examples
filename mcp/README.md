# Databricks MCP (Model Context Protocol) Examples

This directory contains examples of using the Databricks MCP client with Python 3.12.

## Files

- **`databricks_mcp_example.py`** - Python script/notebook that works in both local development and Databricks environments
- **`requirements.txt`** - Python dependencies for MCP functionality

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

## Features

### Authentication Methods
- **OAuth via CLI profiles** - Secure authentication using Databricks CLI
- **Token authentication** - Environment variable based auth for automation
- **Notebook authentication** - Automatic auth in Databricks notebooks

### MCP Client Functionality
- **Tool Discovery** - Lists all available MCP tools
- **Tool Execution** - Demonstrates calling the `system__ai__python_exec` tool
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

### Custom Tool Calls

```python
# Example: Custom SQL execution
result = mcp_client.call_tool(
    "system__ai__sql_exec", 
    {"query": "SELECT COUNT(*) FROM your_table"}
)
print(result.content)
```

### Exploring Available Tools

```python
tools = mcp_client.list_tools()
for tool in tools:
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Schema: {tool.inputSchema}")
```

## Next Steps

1. **Explore other MCP tools** available in your workspace
2. **Integrate MCP calls** into your agent workflows
3. **Build custom MCP servers** for your specific use cases
4. **Combine with DSPy** agents for enhanced functionality

For more information, see the [Databricks MCP documentation](https://docs.databricks.com/en/dev-tools/mcp.html).