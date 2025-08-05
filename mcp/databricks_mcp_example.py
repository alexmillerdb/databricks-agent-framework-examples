# Databricks notebook source
# MAGIC %md
# MAGIC # Databricks MCP Example
# MAGIC 
# MAGIC This demonstrates using the Databricks MCP (Model Context Protocol) client to interact with Unity Catalog functions.
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC - Databricks workspace with Unity Catalog enabled
# MAGIC - Appropriate permissions to access Unity Catalog functions

# COMMAND ----------

# MAGIC %pip install -U "mcp>=1.9" "databricks-sdk[openai]" "mlflow>=3.1.0" "databricks-agents>=1.0.0" "databricks-mcp"

# COMMAND ----------

import os
from typing import Optional

from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

# Import shared authentication utilities
from auth import setup_workspace_client, get_current_user, get_mcp_server_url

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def setup_workspace_client(profile: Optional[str] = None) -> WorkspaceClient:
    """
    Set up WorkspaceClient for authentication.
    
    Args:
        profile: Optional Databricks CLI profile name
        
    Returns:
        Configured WorkspaceClient instance
    """
    # Load environment variables if available
    if load_dotenv:
        load_dotenv()
    
    # Check authentication method
    if os.getenv('DATABRICKS_TOKEN') and os.getenv('DATABRICKS_HOST'):
        print("🏠 Local Development Mode (Token Auth)")
        print("=" * 50)
        print(f"✅ Host: {os.getenv('DATABRICKS_HOST')}")
        print(f"✅ Using token authentication")
        # Token auth from environment
        return WorkspaceClient()
    elif profile:
        print(f"🔐 Using Databricks CLI profile: {profile}")
        return WorkspaceClient(profile=profile)
    else:
        print("☁️  Databricks Environment Mode")
        print("=" * 40)
        print("ℹ️  Using default authentication")
        # Default auth (works in Databricks notebooks)
        return WorkspaceClient()

def get_current_user(workspace_client: WorkspaceClient) -> str:
    """Get current user from workspace."""
    try:
        if is_databricks_notebook():
            # In notebook, try to use spark context if available
            try:
                return spark.sql("SELECT current_user()").collect()[0][0]  # type: ignore
            except NameError:
                # spark not available, fall back to workspace API
                pass
        
        # Use workspace API (works in all environments)
        current_user = workspace_client.current_user.me()
        return current_user.user_name
    except Exception as e:
        print(f"⚠️  Could not get user: {e}")
        return "unknown_user"

def is_databricks_notebook():
    """Check if we're running in a Databricks notebook."""
    try:
        dbutils  # type: ignore
        return True
    except NameError:
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Main MCP Example

# COMMAND ----------

def test_mcp_connection(profile: Optional[str] = None):
    """Test connecting to the MCP server and calling tools.
    
    Args:
        profile: Optional Databricks CLI profile name for authentication
    """
    print("🔌 Testing MCP Server Connection...")
    print("=" * 50)
    
    # Set up workspace client with authentication
    workspace_client = setup_workspace_client(profile)
    
    # Get current user
    user_name = get_current_user(workspace_client)
    print(f"\n👤 User: {user_name}")
    
    # Build MCP server URL
    mcp_server_url = get_mcp_server_url(workspace_client, "system/ai")
    
    print(f"🌐 Connecting to: {mcp_server_url}")
    
    # Create MCP client
    mcp_client = DatabricksMCPClient(
        server_url=mcp_server_url, 
        workspace_client=workspace_client
    )
    
    # List available tools
    print("\n🔍 Discovering available tools...")
    tools = mcp_client.list_tools()
    
    tool_names = [t.name for t in tools]
    print(f"✅ Found {len(tool_names)} tools: {tool_names}")
    
    # Test a tool call
    if "system__ai__python_exec" in tool_names:
        print("\n🧪 Testing system__ai__python_exec tool...")
        
        test_code = """
import datetime
import sys
print(f"Hello from MCP! Current time: {datetime.datetime.now()}")
print(f"Python version: {sys.version}")
print("MCP tool execution successful! 🎉")
"""
        
        result = mcp_client.call_tool(
            "system__ai__python_exec", 
            {"code": test_code}
        )
        print(f"✅ Tool execution result:\n{result.content}")
    else:
        print("⚠️  system__ai__python_exec tool not found")

# COMMAND ----------

def explore_tools(profile: Optional[str] = None):
    """Explore all available MCP tools and their details.
    
    Args:
        profile: Optional Databricks CLI profile name for authentication
    """
    workspace_client = setup_workspace_client(profile)
    mcp_server_url = get_mcp_server_url(workspace_client, "system/ai")
    mcp_client = DatabricksMCPClient(
        server_url=mcp_server_url,
        workspace_client=workspace_client
    )
    
    tools = mcp_client.list_tools()
    
    print("🔧 Available MCP Tools:")
    print("=" * 50)
    
    for tool in tools:
        print(f"\n📋 Tool: {tool.name}")
        if hasattr(tool, 'description') and tool.description:
            print(f"   Description: {tool.description}")
        if hasattr(tool, 'inputSchema') and tool.inputSchema:
            print(f"   Input Schema: {tool.inputSchema}")

# COMMAND ----------

def main(profile: Optional[str] = None):
    """Main function to run the MCP example.
    
    Args:
        profile: Optional Databricks CLI profile name for authentication
            If not provided, will use environment variables or default auth
    """
    print("🚀 Databricks MCP Example")
    print("=" * 50)
    test_mcp_connection(profile)
    
    # Uncomment to explore all tools:
    # print("\n" + "=" * 50)
    # explore_tools(profile)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Example

# COMMAND ----------

# Run when executed as a script or in a notebook
if __name__ == "__main__":
    # For command line usage, you can pass profile as argument
    import argparse
    parser = argparse.ArgumentParser(description='Databricks MCP Example')
    parser.add_argument('--profile', type=str, help='Databricks CLI profile name')
    args = parser.parse_args()
    main(profile=args.profile)
elif is_databricks_notebook():
    # Auto-run in Databricks notebooks
    main()