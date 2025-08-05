#!/usr/bin/env python3
"""
Unity Catalog Functions MCP Example

This example demonstrates how to use the Databricks MCP client with UC Functions.
"""

import argparse
from auth import create_mcp_client_with_auth

def main():
    parser = argparse.ArgumentParser(description='UC Functions MCP Example')
    parser.add_argument('--profile', type=str, help='Databricks CLI profile name')
    parser.add_argument('--catalog', type=str, required=True, help='UC Functions catalog name')
    parser.add_argument('--schema', type=str, required=True, help='UC Functions schema name')
    args = parser.parse_args()
    
    print("🔧 Unity Catalog Functions MCP Example")
    print("=" * 45)
    
    try:
        # Create MCP client for UC Functions
        mcp_client, workspace_client = create_mcp_client_with_auth(
            profile=args.profile,
            server_type="functions",
            catalog=args.catalog,
            schema=args.schema
        )
        
        print(f"✅ Connected to UC Functions MCP server")
        print(f"   Catalog: {args.catalog}")
        print(f"   Schema: {args.schema}")
        
        # List available tools (UC functions)
        tools = mcp_client.list_tools()
        tool_names = [t.name for t in tools]
        print(f"\n🔧 Available UC functions: {len(tool_names)}")
        
        for i, tool in enumerate(tools):
            print(f"   {i+1}. {tool.name}")
            if hasattr(tool, 'description') and tool.description:
                print(f"      Description: {tool.description}")
        
        # Example: Call a function (if available)
        if tool_names:
            print(f"\n💡 To call a function, use:")
            print(f"   result = mcp_client.call_tool('{tool_names[0]}', {{arguments}})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())