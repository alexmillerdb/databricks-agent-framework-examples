#!/usr/bin/env python3
"""
Genie Space MCP Example

This example demonstrates how to use the Databricks MCP client with Genie Space.
"""

import argparse
from auth import create_mcp_client_with_auth

def main():
    parser = argparse.ArgumentParser(description='Genie Space MCP Example')
    parser.add_argument('--profile', type=str, help='Databricks CLI profile name')
    parser.add_argument('--genie-space-id', type=str, required=True, help='Genie space ID')
    args = parser.parse_args()
    
    print("🧞 Genie Space MCP Example")
    print("=" * 30)
    
    try:
        # Create MCP client for Genie Space
        mcp_client, workspace_client = create_mcp_client_with_auth(
            profile=args.profile,
            server_type="genie",
            genie_space_id=args.genie_space_id
        )
        
        print(f"✅ Connected to Genie Space MCP server")
        print(f"   Genie Space ID: {args.genie_space_id}")
        
        # List available tools
        tools = mcp_client.list_tools()
        tool_names = [t.name for t in tools]
        print(f"\n🔧 Available Genie tools: {len(tool_names)}")
        
        for i, tool in enumerate(tools):
            print(f"   {i+1}. {tool.name}")
            if hasattr(tool, 'description') and tool.description:
                print(f"      Description: {tool.description}")
        
        # Example: Ask Genie a question (if chat tool is available)
        chat_tools = [t for t in tool_names if 'chat' in t.lower() or 'ask' in t.lower()]
        if chat_tools:
            print(f"\n💡 Available chat tools: {chat_tools}")
            print(f"   Example usage:")
            print(f"   result = mcp_client.call_tool('{chat_tools[0]}', {{'question': 'What is our revenue trend?'}})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())