"""
Deploy MCP ResponsesAgent to Databricks Model Serving.

This follows the exact deployment pattern from Databricks MCP documentation.
"""

import os
import argparse
from databricks.sdk import WorkspaceClient
from databricks import agents
import mlflow
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from databricks_mcp import DatabricksMCPClient


def main():
    """Deploy MCP ResponsesAgent following Databricks documentation pattern."""
    parser = argparse.ArgumentParser(description='Deploy MCP ResponsesAgent')
    parser.add_argument('--profile', type=str, help='Databricks CLI profile name')
    parser.add_argument('--catalog', type=str, default='users', help='UC catalog for model registration')
    parser.add_argument('--schema', type=str, default='alex_miller', help='UC schema for model registration') 
    parser.add_argument('--model-name', type=str, default='mcp_responses_agent', help='Model name')
    parser.add_argument('--skip-deploy', action='store_true', help='Skip deployment to serving endpoint')
    args = parser.parse_args()
    
    print("🚀 Deploying MCP ResponsesAgent")
    print("=" * 50)
    
    # Configure authentication and MLflow
    databricks_cli_profile = args.profile
    workspace_client = WorkspaceClient(profile=databricks_cli_profile) if databricks_cli_profile else WorkspaceClient()
    current_user = workspace_client.current_user.me().user_name
    
    if databricks_cli_profile:
        mlflow.set_tracking_uri(f"databricks://{databricks_cli_profile}")
        mlflow.set_registry_uri(f"databricks-uc://{databricks_cli_profile}")
        os.environ["DATABRICKS_CONFIG_PROFILE"] = databricks_cli_profile
    else:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    
    # Set experiment
    experiment_name = f"/Users/{current_user}/mcp_responses_agent_experiments"
    mlflow.set_experiment(experiment_name)
    
    print(f"👤 User: {current_user}")
    print(f"🧪 Experiment: {experiment_name}")
    
    # Get LLM endpoint from mcp_responses_agent
    from mcp_responses_agent import LLM_ENDPOINT_NAME
    
    # Define base resources
    resources = [
        DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
        DatabricksFunction("system.ai.python_exec"),
    ]
    
    print(f"🤖 LLM Endpoint: {LLM_ENDPOINT_NAME}")
    
    # Auto-discover resources from managed MCP servers
    host = workspace_client.config.host
    managed_mcp_server_urls = [
        f"{host}/api/2.0/mcp/functions/system/ai",
        # Add more servers as needed:
        # f"{host}/api/2.0/mcp/vector-search/catalog/schema",
        # f"{host}/api/2.0/mcp/functions/catalog/schema", 
        # f"{host}/api/2.0/mcp/genie/space_id",
    ]
    
    print(f"🔍 Discovering resources from MCP servers...")
    for mcp_server_url in managed_mcp_server_urls:
        try:
            print(f"   Checking: {mcp_server_url}")
            mcp_client = DatabricksMCPClient(server_url=mcp_server_url, workspace_client=workspace_client)
            mcp_resources = mcp_client.get_databricks_resources()
            resources.extend(mcp_resources)
            print(f"   ✅ Added {len(mcp_resources)} resources")
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to get resources from {mcp_server_url}: {e}")
    
    print(f"📦 Total resources: {len(resources)}")
    
    # Log and register model
    uc_model_name = f"{args.catalog}.{args.schema}.{args.model_name}"
    
    print(f"📝 Logging model: {uc_model_name}")
    
    with mlflow.start_run(run_name="mcp_responses_agent_deployment"):
        logged_model_info = mlflow.pyfunc.log_model(
            artifact_path="mcp_responses_agent",
            python_model="mcp_responses_agent.py",  # The ResponsesAgent implementation
            resources=resources,
            pip_requirements=[
                "databricks-mcp",
                "databricks-sdk[openai]",
                "mlflow[databricks]>=3.1.0",
                "databricks-agents>=1.0.0",
                "python-dotenv",
            ],
            input_example={
                "input": [{"role": "user", "content": "What's the 100th Fibonacci number?"}]
            },
            code_paths=["mcp_agent/"]  # Include the mcp_agent package
        )
        
        # Log deployment parameters
        mlflow.log_param("llm_endpoint", LLM_ENDPOINT_NAME)
        mlflow.log_param("deployment_type", "mcp_responses_agent")
        mlflow.log_param("mcp_servers", len(managed_mcp_server_urls))
    
    print(f"✅ Model logged: {logged_model_info.model_uri}")
    
    # Register model
    print(f"📋 Registering model: {uc_model_name}")
    registered_model = mlflow.register_model(logged_model_info.model_uri, uc_model_name)
    print(f"✅ Model registered: {uc_model_name} (version {registered_model.version})")
    
    # Deploy to serving (optional)
    if not args.skip_deploy:
        print(f"🚀 Deploying to Model Serving...")
        deployment = agents.deploy(
            model_name=uc_model_name,
            model_version=registered_model.version,
        )
        
        print(f"\n🎉 Deployment Summary:")
        print(f"   Model: {uc_model_name}")
        print(f"   Version: {registered_model.version}")
        print(f"   Endpoint: {deployment.endpoint_name}")
        print(f"   Status: {deployment.state}")
        
        return {
            "model_name": uc_model_name,
            "model_version": registered_model.version,
            "endpoint_name": deployment.endpoint_name,
            "status": deployment.state
        }
    else:
        print(f"\n📝 Model logged and registered successfully:")
        print(f"   Model: {uc_model_name}")
        print(f"   Version: {registered_model.version}")
        print(f"   URI: {logged_model_info.model_uri}")
        print(f"\n💡 To deploy later, run:")
        print(f"   agents.deploy(model_name='{uc_model_name}', model_version='{registered_model.version}')")
        
        return {
            "model_name": uc_model_name,
            "model_version": registered_model.version,
            "model_uri": logged_model_info.model_uri
        }


if __name__ == "__main__":
    try:
        result = main()
        print(f"\n✅ Deployment script completed successfully!")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        raise