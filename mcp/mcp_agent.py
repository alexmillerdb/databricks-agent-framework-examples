"""
Databricks MCP Agent

A production-ready MCP agent that integrates with Databricks workspaces
using OAuth/WorkspaceClient authentication and provides MCP tool access.
"""

import asyncio
from typing import Optional, List, Dict, Any
import logging

from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

# Import shared authentication utilities
from auth import (
    setup_workspace_client,
    get_current_user,
    get_mcp_server_url,
    create_mcp_client_with_auth
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabricksMCPAgent:
    """
    A production MCP agent for Databricks workspaces.
    
    Features:
    - Multiple authentication methods (OAuth, token, notebook)
    - Tool discovery and execution
    - Error handling and logging
    - Async support for tool operations
    """
    
    def __init__(
        self, 
        profile: Optional[str] = None, 
        server_type: str = "system/ai",
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        genie_space_id: Optional[str] = None
    ):
        """
        Initialize the MCP agent.
        
        Args:
            profile: Optional Databricks CLI profile name
            server_type: MCP server type. Options:
                - "system/ai" (default): System AI functions
                - "vector-search": Vector Search MCP server
                - "functions": UC Functions MCP server  
                - "genie": Genie Space MCP server
            catalog: Catalog name (required for vector-search and functions)
            schema: Schema name (required for vector-search and functions)
            genie_space_id: Genie space ID (required for genie)
        """
        self.profile = profile
        self.server_type = server_type
        self.catalog = catalog
        self.schema = schema
        self.genie_space_id = genie_space_id
        self.workspace_client = None
        self.mcp_client = None
        self.available_tools = []

    def connect(self) -> None:
        """
        Connect to the Databricks workspace and MCP server.
        """
        logger.info("🔌 Connecting to Databricks MCP...")
        
        # Set up workspace client using shared auth
        self.workspace_client = setup_workspace_client(self.profile)
        
        # Get current user using shared function
        user_name = get_current_user(self.workspace_client)
        logger.info(f"👤 User: {user_name}")
        
        # Build MCP server URL using shared function
        mcp_server_url = get_mcp_server_url(
            self.workspace_client, 
            self.server_type,
            catalog=self.catalog,
            schema=self.schema,
            genie_space_id=self.genie_space_id
        )
        
        logger.info(f"🌐 Connecting to: {mcp_server_url}")
        
        # Create MCP client
        self.mcp_client = DatabricksMCPClient(
            server_url=mcp_server_url,
            workspace_client=self.workspace_client
        )
        
        # Discover available tools
        self.discover_tools()
        
        logger.info("✅ Connected successfully!")

    def discover_tools(self) -> List[str]:
        """
        Discover available MCP tools.
        
        Returns:
            List of available tool names
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized. Call connect() first.")
        
        logger.info("🔍 Discovering available tools...")
        tools = self.mcp_client.list_tools()
        
        self.available_tools = [t.name for t in tools]
        logger.info(f"✅ Found {len(self.available_tools)} tools: {self.available_tools}")
        
        return self.available_tools

    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information dictionary
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized. Call connect() first.")
        
        tools = self.mcp_client.list_tools()
        
        for tool in tools:
            if tool.name == tool_name:
                return {
                    "name": tool.name,
                    "description": getattr(tool, 'description', None),
                    "inputSchema": getattr(tool, 'inputSchema', None)
                }
        
        raise ValueError(f"Tool '{tool_name}' not found")

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Call an MCP tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        if not self.mcp_client:
            raise RuntimeError("MCP client not initialized. Call connect() first.")
        
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool '{tool_name}' is not available. Available tools: {self.available_tools}")
        
        logger.info(f"🔧 Calling tool: {tool_name}")
        logger.debug(f"Arguments: {arguments}")
        
        try:
            result = self.mcp_client.call_tool(tool_name, arguments)
            logger.info("✅ Tool execution successful")
            return result.content
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {e}")
            raise

    async def call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Asynchronously call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        # Run the synchronous call in an executor to make it async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.call_tool, tool_name, arguments)

    def execute_python_code(self, code: str) -> str:
        """
        Execute Python code using the MCP Python execution tool.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result
        """
        return self.call_tool("system__ai__python_exec", {"code": code})

    def execute_sql_query(self, query: str) -> str:
        """
        Execute SQL query using the MCP SQL execution tool.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Query result
        """
        return self.call_tool("system__ai__sql_exec", {"query": query})

    def search_files(self, search_term: str, limit: int = 10) -> str:
        """
        Search files using the MCP file search tool.
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results
            
        Returns:
            Search results
        """
        return self.call_tool("system__ai__file_search", {
            "search_term": search_term,
            "limit": limit
        })

    def list_available_tools(self) -> List[str]:
        """
        Get list of available tools.
        
        Returns:
            List of tool names
        """
        return self.available_tools.copy()

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information.
        
        Returns:
            Connection information dictionary
        """
        if not self.workspace_client:
            return {"status": "disconnected"}
        
        return {
            "status": "connected",
            "host": self.workspace_client.config.host,
            "user": get_current_user(self.workspace_client),
            "profile": self.profile,
            "server_type": self.server_type,
            "catalog": self.catalog,
            "schema": self.schema,
            "genie_space_id": self.genie_space_id,
            "available_tools": len(self.available_tools)
        }


def create_agent(
    profile: Optional[str] = None, 
    server_type: str = "system/ai",
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    genie_space_id: Optional[str] = None
) -> DatabricksMCPAgent:
    """
    Create and connect a Databricks MCP agent.
    
    Args:
        profile: Optional Databricks CLI profile name
        server_type: MCP server type
        catalog: Catalog name (required for vector-search and functions)
        schema: Schema name (required for vector-search and functions)
        genie_space_id: Genie space ID (required for genie)
        
    Returns:
        Connected MCP agent instance
    """
    agent = DatabricksMCPAgent(
        profile=profile, 
        server_type=server_type,
        catalog=catalog,
        schema=schema,
        genie_space_id=genie_space_id
    )
    agent.connect()
    return agent


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Databricks MCP Agent')
    parser.add_argument('--profile', type=str, help='Databricks CLI profile name')
    parser.add_argument('--server-type', type=str, default='system/ai', 
                       help='MCP server type: system/ai, vector-search, functions, genie')
    parser.add_argument('--catalog', type=str, help='Catalog name (required for vector-search and functions)')
    parser.add_argument('--schema', type=str, help='Schema name (required for vector-search and functions)')
    parser.add_argument('--genie-space-id', type=str, help='Genie space ID (required for genie)')
    parser.add_argument('--test', action='store_true', help='Run test scenarios')
    args = parser.parse_args()
    
    # Create and connect agent
    try:
        agent = create_agent(
            profile=args.profile, 
            server_type=args.server_type,
            catalog=args.catalog,
            schema=args.schema,
            genie_space_id=args.genie_space_id
        )
        
        print(f"\n🎉 MCP Agent connected successfully!")
        print(f"Connection info: {agent.get_connection_info()}")
        
        if args.test:
            print("\n🧪 Running test scenarios...")
            
            # Test 1: Python execution
            if "system__ai__python_exec" in agent.available_tools:
                print("\n📝 Testing Python execution...")
                test_code = """
import datetime
import platform
print(f"Hello from MCP Agent! Current time: {datetime.datetime.now()}")
print(f"Platform: {platform.platform()}")
print("✅ Python execution test successful!")
"""
                result = agent.execute_python_code(test_code)
                print(f"Result:\n{result}")
            
            # Test 2: Tool information
            print(f"\n📋 Available tools: {agent.list_available_tools()}")
            
            # Test 3: Connection info
            print(f"\n🔗 Connection info: {agent.get_connection_info()}")
            
    except Exception as e:
        logger.error(f"❌ Failed to create MCP agent: {e}")
        raise