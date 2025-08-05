"""
Deploy Databricks MCP Agent

This script deploys the MCP agent to Databricks Model Serving following
the same pattern as the DSPy RAG agent deployment.
"""

import os
import argparse
from typing import Optional

import mlflow
from databricks import agents
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from databricks_mcp import DatabricksMCPClient

# Import shared authentication utilities
from auth import setup_workspace_client, setup_mlflow_for_profile, get_mcp_server_url

# Configuration
def get_deployment_config():
    """Get deployment configuration."""
    return {
        "UC_CATALOG": "users",
        "UC_SCHEMA": "alex_miller", 
        "UC_MODEL_NAME": "databricks_mcp_agent",
        "MCP_SERVER_TYPE": "system/ai",
        "LLM_ENDPOINT_NAME": "databricks/databricks-claude-3-7-sonnet"
    }

def setup_environment_and_mlflow(profile: Optional[str] = None):
    """Set up workspace client and MLflow configuration."""
    print("🚀 Setting up environment and MLflow...")
    
    # Set up workspace client with authentication
    workspace_client = setup_workspace_client(profile, verbose=True)
    
    # Configure MLflow for the profile/workspace
    current_user = setup_mlflow_for_profile(workspace_client, profile)
    
    # Set up MLflow experiment
    experiment_name = f"/Users/{current_user}/databricks_mcp_agent_experiments"
    mlflow.set_experiment(experiment_name)
    
    print(f"✅ MLflow experiment: {experiment_name}")
    print(f"✅ Registry URI: {mlflow.get_registry_uri()}")
    
    return workspace_client, current_user

def collect_databricks_resources(workspace_client, config):
    """Collect Databricks resources needed for the MCP agent."""
    print("📦 Collecting Databricks resources...")
    
    resources = [
        DatabricksServingEndpoint(endpoint_name=config["LLM_ENDPOINT_NAME"]),
        DatabricksFunction("system.ai.python_exec"),
        # Add more functions as needed
        # DatabricksFunction("system.ai.sql_exec"),
        # DatabricksFunction("system.ai.file_search"),
    ]
    
    # Get MCP server URL and collect resources
    mcp_server_url = get_mcp_server_url(workspace_client, config["MCP_SERVER_TYPE"])
    print(f"🔗 MCP Server URL: {mcp_server_url}")
    
    try:
        mcp_client = DatabricksMCPClient(server_url=mcp_server_url, workspace_client=workspace_client)
        mcp_resources = mcp_client.get_databricks_resources()
        resources.extend(mcp_resources)
        print(f"✅ Added {len(mcp_resources)} MCP resources")
    except Exception as e:
        print(f"⚠️  Warning: Could not collect MCP resources: {e}")
    
    return resources

def log_and_register_model(workspace_client, config):
    """Log the MCP agent model to MLflow and register it."""
    print("📋 Logging model to MLflow...")
    
    # Get agent script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    agent_script = os.path.join(script_dir, "mcp_agent.py")
    
    if not os.path.exists(agent_script):
        raise FileNotFoundError(f"Agent script not found: {agent_script}")
    
    # Collect resources
    resources = collect_databricks_resources(workspace_client, config)
    
    # Build model name
    model_name = f"{config['UC_CATALOG']}.{config['UC_SCHEMA']}.{config['UC_MODEL_NAME']}"
    
    # Log model
    with mlflow.start_run(run_name="mcp_agent_deployment"):
        logged_model_info = mlflow.pyfunc.log_model(
            artifact_path="mcp_agent",
            python_model=agent_script,
            resources=resources,
            pip_requirements=[
                "databricks-mcp",
                "databricks-sdk[openai]",
                "mlflow[databricks]>=3.1.0",
                "databricks-agents>=1.0.0",
                "python-dotenv",
            ],
            input_example={
                "messages": [{"role": "user", "content": "What Python libraries are available?"}]
            },
            code_paths=["auth.py"]  # Include shared auth module
        )
        
        # Log parameters
        mlflow.log_param("mcp_server_type", config["MCP_SERVER_TYPE"])
        mlflow.log_param("llm_endpoint", config["LLM_ENDPOINT_NAME"])
        mlflow.log_param("deployment_type", "mcp_agent")
    
    print(f"✅ Model logged: {logged_model_info.model_uri}")
    
    # Register model
    print(f"📝 Registering model: {model_name}")
    registered_model = mlflow.register_model(logged_model_info.model_uri, model_name)
    
    print(f"✅ Model registered: {model_name} (version {registered_model.version})")
    
    return logged_model_info, registered_model, model_name

def deploy_to_serving(model_name, model_version):
    """Deploy the registered model to Databricks Model Serving."""
    print(f"🚀 Deploying {model_name} version {model_version} to Model Serving...")
    
    try:
        deployment = agents.deploy(
            model_name=model_name,
            model_version=model_version,
            scale_to_zero=True  # Enable scale-to-zero for cost optimization
        )
        
        print(f"✅ Deployment completed!")
        print(f"🔗 Endpoint: {deployment.endpoint_name}")
        print(f"📊 Status: {deployment.state}")
        
        return deployment
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        raise

def main():
    """Main deployment workflow."""
    parser = argparse.ArgumentParser(description='Deploy Databricks MCP Agent')
    parser.add_argument('--profile', type=str, help='Databricks CLI profile name')
    parser.add_argument('--skip-deploy', action='store_true', 
                       help='Skip deployment to serving endpoint (log and register only)')
    args = parser.parse_args()
    
    print("🚀 Databricks MCP Agent Deployment")
    print("=" * 50)
    
    try:
        # Setup environment and configuration
        config = get_deployment_config()
        workspace_client, current_user = setup_environment_and_mlflow(args.profile)
        
        print(f"\n📋 Deployment Configuration:")
        print(f"   Catalog: {config['UC_CATALOG']}")
        print(f"   Schema: {config['UC_SCHEMA']}")
        print(f"   Model: {config['UC_MODEL_NAME']}")
        print(f"   MCP Server: {config['MCP_SERVER_TYPE']}")
        print(f"   LLM Endpoint: {config['LLM_ENDPOINT_NAME']}")
        print(f"   User: {current_user}")
        
        # Log and register model
        logged_model_info, registered_model, model_name = log_and_register_model(workspace_client, config)
        
        # Deploy to serving (optional)
        if not args.skip_deploy:
            deployment = deploy_to_serving(model_name, registered_model.version)
            
            print(f"\n🎉 Deployment Summary:")
            print(f"   Model: {model_name}")
            print(f"   Version: {registered_model.version}")
            print(f"   Endpoint: {deployment.endpoint_name}")
        else:
            print(f"\n📝 Model logged and registered successfully:")
            print(f"   Model: {model_name}")
            print(f"   Version: {registered_model.version}")
            print(f"   URI: {logged_model_info.model_uri}")
            print(f"\n💡 To deploy later, run:")
            print(f"   agents.deploy(model_name='{model_name}', model_version='{registered_model.version}')")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        raise

if __name__ == "__main__":
    main()