# Databricks MCP Agent Development Guide

*Based on official Databricks documentation for Managed MCP Servers (Beta feature)*

## Overview

Model Context Protocol (MCP) servers act as bridges that let AI agents access external data and tools. Databricks provides managed MCP servers to instantly connect your agents to data stored in Unity Catalog, vector search indexes, and custom functions.

## Available Managed MCP Servers

Databricks provides three types of managed MCP servers:

| MCP Server | Description | URL Pattern |
|------------|-------------|-------------|
| Vector search | Query vector search indexes | `https://<workspace-hostname>/api/2.0/mcp/vector-search/<schema>/<table>` |
| Unity Catalog functions | Run custom functions | `https://<workspace-hostname>/api/2.0/mcp/functions/<schema>/<function>` |
| Genie space | Query Genie spaces | `https://<workspace-hostname>/api/2.0/mcp/genie/{space_id}` |

**Note:** The Managed MCP server for Genie invokes Genie as an MCP tool without passing history. For history support, consider using [Genie in a multi-agent system](https://docs.databricks.com/aws/en/generative-ai/agent-framework/multi-agent-genie).

## Example Use Case: Customer Support Agent

A customer support agent could connect to multiple managed MCP servers:

- **Vector search**: `https://<workspace-hostname>/api/2.0/mcp/vector-search/prod/customer_support`
  - Searches support tickets and documentation
- **Genie space**: `https://<workspace-hostname>/api/2.0/mcp/genie/{billing_space_id}`
  - Queries billing data and customer information  
- **UC functions**: `https://<workspace-hostname>/api/2.0/mcp/functions/prod/billing`
  - Runs custom functions for account lookups and updates

## Prerequisites

### Environment Setup

1. **Authentication**: Use OAuth to authenticate to your workspace
   ```bash
   databricks auth login --host https://<your-workspace-hostname>
   ```
   Create and remember the profile name when prompted.

2. **Python Environment**: Ensure Python 3.12 or above, then install dependencies:
   ```bash
   pip install -U "mcp>=1.9" "databricks-sdk[openai]" "mlflow>=3.1.0" "databricks-agents>=1.0.0" "databricks-mcp"
   ```

3. **Workspace Requirements**: [Serverless compute](https://docs.databricks.com/aws/en/admin/workspace-settings/serverless) must be enabled in your workspace.

## Development Process

### Step 1: Test Connection

Validate your connection to the MCP server with this test script:

```python
from databricks_mcp import DatabricksMCPClient
from databricks.sdk import WorkspaceClient

# TODO: Update to your Databricks CLI profile name
databricks_cli_profile = "YOUR_DATABRICKS_CLI_PROFILE"
assert (
    databricks_cli_profile != "YOUR_DATABRICKS_CLI_PROFILE"
), "Set databricks_cli_profile to the Databricks CLI profile name you specified when configuring authentication to the workspace"

workspace_client = WorkspaceClient(profile=databricks_cli_profile)
workspace_hostname = workspace_client.config.host
mcp_server_url = f"{workspace_hostname}/api/2.0/mcp/functions/system/ai"

def test_connect_to_server():
    mcp_client = DatabricksMCPClient(server_url=mcp_server_url, workspace_client=workspace_client)
    tools = mcp_client.list_tools()
    print(
        f"Discovered tools {[t.name for t in tools]} "
        f"from MCP server {mcp_server_url}"
    )
    result = mcp_client.call_tool(
        "system__ai__python_exec", {"code": "print('Hello, world!')"}
    )
    print(
        f"Called system__ai__python_exec tool and got result "
        f"{result.content}"
    )

if __name__ == "__main__":
    test_connect_to_server()
```

### Step 2: Create Agent Implementation

Create a file named `mcp_agent.py` with your agent implementation. The agent should:

1. **Configure endpoints and authentication**:
   - Set `LLM_ENDPOINT_NAME` (e.g., `"databricks-claude-3-7-sonnet"`)
   - Set `DATABRICKS_CLI_PROFILE` to your profile name
   - Define `MANAGED_MCP_SERVER_URLS` for managed servers
   - Define `CUSTOM_MCP_SERVER_URLS` for custom servers

2. **Implement core functionality**:
   - Convert between ResponsesAgent message format and ChatCompletions format
   - Fetch tool information from MCP servers
   - Handle tool execution and responses
   - Implement single-turn agent logic

3. **Key Components**:
   - `DatabricksMCPClient` for server communication
   - `ResponsesAgent` base class for agent implementation
   - Tool discovery and execution logic
   - Message format conversion utilities

### Step 3: Local Testing

Run your agent locally:

```python
 import json
 import uuid
 import asyncio
 from typing import Any, Callable, List
 from pydantic import BaseModel

 import mlflow
 from mlflow.pyfunc import ResponsesAgent
 from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

 from databricks_mcp import DatabricksMCPClient
 from databricks.sdk import WorkspaceClient

 # 1) CONFIGURE YOUR ENDPOINTS/PROFILE
 LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
 SYSTEM_PROMPT = "You are a helpful assistant."
 DATABRICKS_CLI_PROFILE = "YOUR_DATABRICKS_CLI_PROFILE"
 assert (
     DATABRICKS_CLI_PROFILE != "YOUR_DATABRICKS_CLI_PROFILE"
 ), "Set DATABRICKS_CLI_PROFILE to the Databricks CLI profile name you specified when configuring authentication to the workspace"
 workspace_client = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)
 host = workspace_client.config.host
 # Add more MCP server URLs here if desired, e.g
 # f"{host}/api/2.0/mcp/vector-search/prod/billing"
 # to include vector search indexes under the prod.billing schema, or
 # f"{host}/api/2.0/mcp/genie/<genie_space_id>"
 # to include a Genie space
 MANAGED_MCP_SERVER_URLS = [
     f"{host}/api/2.0/mcp/functions/system/ai",
 ]
 # Add Custom MCP Servers hosted on Databricks Apps
 CUSTOM_MCP_SERVER_URLS = []



 # 2) HELPER: convert between ResponsesAgent “message dict” and ChatCompletions format
 def _to_chat_messages(msg: dict[str, Any]) -> List[dict]:
     """
     Take a single ResponsesAgent‐style dict and turn it into one or more
     ChatCompletions‐compatible dict entries.
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


 # 3) “MCP SESSION” + TOOL‐INVOCATION LOGIC
 def _make_exec_fn(
     server_url: str, tool_name: str, ws: WorkspaceClient
 ) -> Callable[..., str]:
     def exec_fn(**kwargs):
         mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=ws)
         response = mcp_client.call_tool(tool_name, kwargs)
         return "".join([c.text for c in response.content])

     return exec_fn


 class ToolInfo(BaseModel):
     name: str
     spec: dict
     exec_fn: Callable


 def _fetch_tool_infos(ws: WorkspaceClient, server_url: str) -> List[ToolInfo]:
     print(f"Listing tools from MCP server {server_url}")
     infos: List[ToolInfo] = []
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
                 name=t.name, spec=spec, exec_fn=_make_exec_fn(server_url, t.name, ws)
             )
         )
     return infos


 # 4) “SINGLE‐TURN” AGENT CLASS
 class SingleTurnMCPAgent(ResponsesAgent):
     def _call_llm(self, history: List[dict], ws: WorkspaceClient, tool_infos):
         """
         Send current history → LLM, returning the raw response dict.
         """
         client = ws.serving_endpoints.get_open_ai_client()
         flat_msgs = []
         for msg in history:
             flat_msgs.extend(_to_chat_messages(msg))
         return client.chat.completions.create(
             model=LLM_ENDPOINT_NAME,
             messages=flat_msgs,
             tools=[ti.spec for ti in tool_infos],
         )

     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
         ws = WorkspaceClient(profile=DATABRICKS_CLI_PROFILE)

         # 1) build initial history: system + user
         history: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
         for inp in request.input:
             history.append(inp.model_dump())

         # 2) call LLM once
         tool_infos = [
             tool_info
             for mcp_server_url in (MANAGED_MCP_SERVER_URLS + CUSTOM_MCP_SERVER_URLS)
             for tool_info in _fetch_tool_infos(ws, mcp_server_url)
         ]
         tools_dict = {tool_info.name: tool_info for tool_info in tool_infos}
         llm_resp = self._call_llm(history, ws, tool_infos)
         raw_choice = llm_resp.choices[0].message.to_dict()
         raw_choice["id"] = uuid.uuid4().hex
         history.append(raw_choice)

         tool_calls = raw_choice.get("tool_calls") or []
         if tool_calls:
             # (we only support a single tool in this “single‐turn” example)
             fc = tool_calls[0]
             name = fc["function"]["name"]
             args = json.loads(fc["function"]["arguments"])
             try:
                 tool_info = tools_dict[name]
                 result = tool_info.exec_fn(**args)
             except Exception as e:
                 result = f"Error invoking {name}: {e}"

             # 4) append the “tool” output
             history.append(
                 {
                     "type": "function_call_output",
                     "role": "tool",
                     "id": uuid.uuid4().hex,
                     "tool_call_id": fc["id"],
                     "output": result,
                 }
             )

             # 5) call LLM a second time and treat that reply as final
             followup = (
                 self._call_llm(history, ws, tool_infos=[]).choices[0].message.to_dict()
             )
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

         # 6) if no tool_calls at all, return the assistant’s original reply
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


 mlflow.models.set_model(SingleTurnMCPAgent())

 if __name__ == "__main__":
     req = ResponsesAgentRequest(
         input=[{"role": "user", "content": "What's the 100th Fibonacci number?"}]
     )
     resp = SingleTurnMCPAgent().predict(req)
     for item in resp.output:
         print(item)
```

## Deployment

### Resource Requirements

When deploying, you must specify all resources your agent needs access to at logging time. For example, if your agent uses:

- `https://<workspace-hostname>/api/2.0/mcp/vector-search/prod/customer_support`
- `https://<workspace-hostname>/api/2.0/mcp/vector-search/prod/billing`  
- `https://<workspace-hostname>/api/2.0/mcp/functions/prod/billing`

You must specify:
- All vector search indexes in `prod.customer_support` and `prod.billing` schemas
- All Unity Catalog functions in `prod.billing`

### Deployment Script

Use this pattern for deployment:

```python
import os
from databricks.sdk import WorkspaceClient
from databricks import agents
import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mcp_agent import LLM_ENDPOINT_NAME
from databricks_mcp import DatabricksMCPClient

# Configure authentication and MLflow
databricks_cli_profile = "YOUR_DATABRICKS_CLI_PROFILE"
workspace_client = WorkspaceClient(profile=databricks_cli_profile)
current_user = workspace_client.current_user.me().user_name

mlflow.set_tracking_uri(f"databricks://{databricks_cli_profile}")
mlflow.set_registry_uri(f"databricks-uc://{databricks_cli_profile}")
mlflow.set_experiment(f"/Users/{current_user}/databricks_docs_example_mcp_agent")
os.environ["DATABRICKS_CONFIG_PROFILE"] = databricks_cli_profile

# Define resources
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksFunction("system.ai.python_exec"),
    # Add custom apps with: DatabricksApp(app_name="app-name")
]

# Auto-discover resources from managed MCP servers
for mcp_server_url in MANAGED_MCP_SERVER_URLS:
    mcp_client = DatabricksMCPClient(server_url=mcp_server_url, workspace_client=workspace_client)
    resources.extend(mcp_client.get_databricks_resources())

# Log and deploy
with mlflow.start_run():
    logged_model_info = mlflow.pyfunc.log_model(
        artifact_path="mcp_agent",
        python_model="mcp_agent.py",  # Path to your agent script
        resources=resources,
    )

UC_MODEL_NAME = "main.default.databricks_docs_mcp_agent"
registered_model = mlflow.register_model(logged_model_info.model_uri, UC_MODEL_NAME)

agents.deploy(
    model_name=UC_MODEL_NAME,
    model_version=registered_model.version,
)
```

## Key Libraries and Classes

- **`databricks-mcp`**: Python library for simplified authentication
- **`DatabricksMCPClient`**: Main client for interacting with MCP servers
  - `list_tools()`: Discover available tools
  - `call_tool(name, kwargs)`: Execute tools
  - `get_databricks_resources()`: Auto-discover required resources
- **`ResponsesAgent`**: Base class for agent implementation
- **`WorkspaceClient`**: Databricks SDK client for workspace operations

## Important Notes

- This feature is currently in **Beta**
- Databricks MCP servers are secure by default and require authentication
- Use the `databricks-mcp` library to simplify authentication in custom agent code
- Follow the [standard agent deployment process](https://docs.databricks.com/aws/en/generative-ai/agent-framework/deploy-agent) when ready to deploy
- For custom MCP servers hosted on Databricks Apps, configure authorization by explicitly including the server as a resource when logging your model

## Next Steps

After creating your agent, you can [connect external services](https://docs.databricks.com/aws/en/generative-ai/mcp/connect-external-services) like Cursor and Claude Desktop to managed MCP servers.