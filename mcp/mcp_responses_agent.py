"""
MCP ResponsesAgent implementation following Databricks documentation exactly.

This implements the proper ResponsesAgent base class for deployment with Databricks Agents Framework.
"""

import json
import uuid
import os
from typing import Any, Callable, List, Dict, Optional
from pydantic import BaseModel

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient
from mcp_agent import setup_workspace_client

# Configuration - can be overridden via environment variables
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-3-7-sonnet")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant with access to various tools and data sources.")
DATABRICKS_CLI_PROFILE = os.getenv("DATABRICKS_CLI_PROFILE")

# MCP Server URLs - configure based on your needs
def get_mcp_server_urls() -> List[str]:
    """Get MCP server URLs from environment or defaults."""
    try:
        # Try to get workspace client to build URLs
        workspace_client = setup_workspace_client(DATABRICKS_CLI_PROFILE, verbose=False)
        host = workspace_client.config.host
        
        # Default to system AI functions
        default_urls = [f"{host}/api/2.0/mcp/functions/system/ai"]
        
        # Add additional servers from environment if specified
        custom_servers = os.getenv("MCP_SERVER_URLS", "").split(",")
        custom_servers = [url.strip() for url in custom_servers if url.strip()]
        
        return default_urls + custom_servers
        
    except Exception:
        # Fallback for testing - will be replaced at deployment time
        return ["https://placeholder/api/2.0/mcp/functions/system/ai"]


# Message format conversion utilities (from Databricks documentation)
def _to_chat_messages(msg: dict[str, Any]) -> List[dict]:
    """
    Convert ResponsesAgent message format to ChatCompletions format.
    
    This follows the exact pattern from Databricks documentation.
    """
    msg_type = msg.get("type")
    
    if msg_type == "function_call":
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": msg["call_id"],
                        "type": "function",
                        "function": {
                            "name": msg["name"],
                            "arguments": msg["arguments"],
                        },
                    }
                ],
            }
        ]
    elif msg_type == "message" and isinstance(msg["content"], list):
        return [
            {
                "role": "assistant" if msg["role"] == "assistant" else msg["role"],
                "content": content["text"],
            }
            for content in msg["content"]
        ]
    elif msg_type == "function_call_output":
        return [
            {
                "role": "tool",
                "content": msg["output"],
                "tool_call_id": msg["tool_call_id"],
            }
        ]
    else:
        # fallback for plain {"role": ..., "content": "..."} or similar
        return [
            {
                k: v
                for k, v in msg.items()
                if k in ("role", "content", "name", "tool_calls", "tool_call_id")
            }
        ]


# Tool execution utilities
def _make_exec_fn(server_url: str, tool_name: str, ws: WorkspaceClient) -> Callable[..., str]:
    """Create tool execution function for a specific MCP tool."""
    def exec_fn(**kwargs):
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
        response = mcp_client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in response.content])
    
    return exec_fn


class ToolInfo(BaseModel):
    """Information about an available MCP tool."""
    name: str
    spec: dict
    exec_fn: Callable


def _fetch_tool_infos(ws: WorkspaceClient, server_url: str) -> List[ToolInfo]:
    """
    Fetch tool information from an MCP server.
    
    This follows the exact pattern from Databricks documentation.
    """
    print(f"Listing tools from MCP server {server_url}")
    infos: List[ToolInfo] = []
    
    try:
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
        mcp_tools = mcp_client.list_tools()
        
        for t in mcp_tools:
            schema = t.inputSchema.copy()
            if "properties" not in schema:
                schema["properties"] = {}
            
            spec = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": schema,
                },
            }
            
            infos.append(
                ToolInfo(
                    name=t.name, 
                    spec=spec, 
                    exec_fn=_make_exec_fn(server_url, t.name, ws)
                )
            )
    except Exception as e:
        print(f"Warning: Failed to fetch tools from {server_url}: {e}")
    
    return infos


class MCPResponsesAgent(ResponsesAgent):
    """
    MCP ResponsesAgent implementation following Databricks documentation.
    
    This agent connects to MCP servers and provides tool execution capabilities
    through the proper ResponsesAgent interface.
    """
    
    def _call_llm(self, history: List[dict], ws: WorkspaceClient, tool_infos: List[ToolInfo]) -> Any:
        """
        Call LLM with current conversation history and available tools.
        
        Args:
            history: Conversation history in ChatCompletions format
            ws: WorkspaceClient for authentication
            tool_infos: Available tool information
            
        Returns:
            LLM response object
        """
        client = ws.serving_endpoints.get_open_ai_client()
        
        # Convert all messages to flat ChatCompletions format
        flat_msgs = []
        for msg in history:
            flat_msgs.extend(_to_chat_messages(msg))
        
        # Call LLM with tools
        tools_param = [ti.spec for ti in tool_infos] if tool_infos else None
        
        # Don't pass empty tools list
        if tools_param:
            return client.chat.completions.create(
                model=LLM_ENDPOINT_NAME,
                messages=flat_msgs,
                tools=tools_param,
            )
        else:
            return client.chat.completions.create(
                model=LLM_ENDPOINT_NAME,
                messages=flat_msgs,
            )
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Main prediction method implementing single-turn agent pattern.
        
        This follows the exact single-turn pattern from Databricks documentation:
        1. Call LLM with tools
        2. Execute any tool calls
        3. Call LLM again with results
        4. Return final response
        """
        # Set up workspace client
        ws = setup_workspace_client(DATABRICKS_CLI_PROFILE, verbose=False)
        
        # Build initial conversation history: system + user messages
        history: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for inp in request.input:
            history.append(inp.model_dump())
        
        # Discover available tools from all MCP servers
        tool_infos = []
        mcp_server_urls = get_mcp_server_urls()
        
        for mcp_server_url in mcp_server_urls:
            server_tools = _fetch_tool_infos(ws, mcp_server_url)
            tool_infos.extend(server_tools)
        
        # Create tools dictionary for quick lookup
        tools_dict = {tool_info.name: tool_info for tool_info in tool_infos}
        
        # Step 1: Call LLM with available tools
        llm_resp = self._call_llm(history, ws, tool_infos)
        raw_choice = llm_resp.choices[0].message.to_dict()
        raw_choice["id"] = uuid.uuid4().hex
        history.append(raw_choice)
        
        # Step 2: Check if LLM wants to call tools
        tool_calls = raw_choice.get("tool_calls") or []
        
        if tool_calls:
            # Execute the first tool call (single-turn pattern)
            fc = tool_calls[0]
            name = fc["function"]["name"]
            
            try:
                args = json.loads(fc["function"]["arguments"])
                tool_info = tools_dict[name]
                result = tool_info.exec_fn(**args)
            except Exception as e:
                result = f"Error invoking {name}: {e}"
            
            # Step 3: Add tool output to history
            history.append({
                "type": "function_call_output",
                "role": "tool",
                "id": uuid.uuid4().hex,
                "tool_call_id": fc["id"],
                "output": result,
            })
            
            # Step 4: Call LLM again with tool results (no tools this time)
            followup = self._call_llm(history, ws, []).choices[0].message.to_dict()
            followup["id"] = uuid.uuid4().hex
            
            assistant_text = followup.get("content", "")
            return ResponsesAgentResponse(
                output=[
                    {
                        "id": uuid.uuid4().hex,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": assistant_text}],
                    }
                ],
                custom_outputs=request.custom_inputs,
            )
        
        # Step 5: No tool calls - return direct response
        assistant_text = raw_choice.get("content", "")
        return ResponsesAgentResponse(
            output=[
                {
                    "id": uuid.uuid4().hex,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": assistant_text}],
                }
            ],
            custom_outputs=request.custom_inputs,
        )


# Required for MLflow deployment
mlflow.models.set_model(MCPResponsesAgent())


if __name__ == "__main__":
    # Test the agent locally
    req = ResponsesAgentRequest(
        input=[{"role": "user", "content": "What's the 10th Fibonacci number?"}]
    )
    
    agent = MCPResponsesAgent()
    resp = agent.predict(req)
    
    print("🚀 Test Response:")
    for item in resp.output:
        print(item)