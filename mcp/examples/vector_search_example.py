#!/usr/bin/env python3
"""
Vector Search MCP Example

This example demonstrates how to use the Databricks MCP client with Vector Search.
"""

import argparse
from auth import create_mcp_client_with_auth

def main():
    parser = argparse.ArgumentParser(description='Vector Search MCP Example')
    parser.add_argument('--profile', type=str, help='Databricks CLI profile name')
    parser.add_argument('--catalog', type=str, required=True, help='Vector Search catalog name')
    parser.add_argument('--schema', type=str, required=True, help='Vector Search schema name')
    args = parser.parse_args()
    
    print("🔍 Vector Search MCP Example")
    print("=" * 40)
    
    try:
        # Create MCP client for Vector Search
        mcp_client, workspace_client = create_mcp_client_with_auth(
            profile=args.profile,
            server_type="vector-search",
            catalog=args.catalog,
            schema=args.schema
        )
        
        print(f"✅ Connected to Vector Search MCP server")
        print(f"   Catalog: {args.catalog}")
        print(f"   Schema: {args.schema}")
        
        # List available tools
        tools = mcp_client.list_tools()
        tool_names = [t.name for t in tools]
        print(f"\n🔧 Available tools: {tool_names}")
        
        # Example: Search vectors (if available)
        if tool_names:
            print(f"\n📋 First tool details:")
            first_tool = tools[0]
            print(f"   Name: {first_tool.name}")
            if hasattr(first_tool, 'description'):
                print(f"   Description: {first_tool.description}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())