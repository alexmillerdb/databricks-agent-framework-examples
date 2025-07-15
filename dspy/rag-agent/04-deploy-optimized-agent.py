# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Optimized DSPy RAG Agent
# MAGIC 
# MAGIC This notebook deploys the optimized DSPy RAG agent to Databricks Model Serving
# MAGIC using MLflow ModelConfig for parameterized deployment.

# COMMAND ----------
%pip install -qqq --upgrade "mlflow[databricks]>=3.1" dspy-ai databricks-agents openai uv
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration Setup
# MAGIC 
# MAGIC Load optimized configuration using MLflow ModelConfig.

# COMMAND ----------
import uuid, os, mlflow, dspy
from pkg_resources import get_distribution
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatAgentChunk

# Load optimized configuration
config_path = "config_optimized.yaml"
model_config = mlflow.models.ModelConfig(development_config=config_path)

# Configuration from ModelConfig
catalog = "users"
schema = "alex_miller"
vector_search_index_name = "wikipedia_chunks_index"
UC_VS_INDEX_NAME = f"{catalog}.{schema}.{vector_search_index_name}"

# Get model name from config
mlflow_config = model_config.get("mlflow_config") or {}
model_name = mlflow_config.get("registered_model_name", "dspy_rag_agent_optimized")
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# Set environment variables from config
llm_config = model_config.get("llm_config") or {}
os.environ["VS_INDEX_FULLNAME"] = UC_VS_INDEX_NAME
os.environ["DSPY_LLM_ENDPOINT"] = llm_config.get("endpoint", "databricks/databricks-claude-3-7-sonnet")

print(f"Using LLM endpoint: {os.environ['DSPY_LLM_ENDPOINT']}")
print(f"Using Vector Search index: {UC_VS_INDEX_NAME}")
print(f"Model name: {UC_MODEL_NAME}")
print(f"Configuration loaded from: {config_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## MLflow Configuration

# COMMAND ----------
# MLflow configuration
mlflow.set_tracking_uri("databricks")
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Set experiment based on config
experiment_name = f"{mlflow_config.get('experiment_name', 'dspy_rag_agent')}_optimized_deployment"
mlflow.set_experiment(f"/Users/{user_name}/{experiment_name}")

# Configure tracing based on config
agent_config = model_config.get("agent_config") or {}
if agent_config.get("enable_tracing", False):
    mlflow.dspy.autolog()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Download Optimized Program

# COMMAND ----------
def download_optimized_program():
    """Download the optimized program from MLflow artifacts."""
    
    # You can get this run_id from the compilation notebook
    # Or search for the latest compilation run
    client = mlflow.tracking.MlflowClient()
    
    # Get experiment name from config
    optimization_experiment_name = f"{mlflow_config.get('experiment_name', 'dspy_rag_agent')}_optimization"
    experiment = mlflow.get_experiment_by_name(f"/Users/{user_name}/{optimization_experiment_name}")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No completed compilation runs found in experiment: {optimization_experiment_name}")
    
    latest_run = runs[0]
    
    # Download the optimized program
    artifact_path = client.download_artifacts(
        latest_run.info.run_id, 
        "optimized_program/optimized_rag_program.json"
    )
    
    # Set the path for the agent to use
    os.environ["DSPY_OPTIMIZED_PROGRAM_PATH"] = artifact_path
    
    print(f"Downloaded optimized program from run: {latest_run.info.run_id}")
    print(f"Optimized program path: {artifact_path}")
    return artifact_path

# COMMAND ----------
# Download the optimized program
try:
    optimized_program_path = download_optimized_program()
except Exception as e:
    print(f"Error downloading optimized program: {e}")
    print("Falling back to using non-optimized program")
    optimized_program_path = None

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test the Optimized Agent

# COMMAND ----------
from agent import DSPyRAGChatAgent, LLM_ENDPOINT

# Create agent with optimized program and config
if optimized_program_path:
    print("Testing optimized agent with downloaded program...")
    optimized_agent = DSPyRAGChatAgent(config=model_config)
else:
    print("Testing agent with default configuration...")
    optimized_agent = DSPyRAGChatAgent(config=model_config)

# Test the agent
test_questions = [
    "Who is Zeus?",
    "What is Greek mythology?",
    "Who is Frederick Barbarossa?"
]

print("\nTesting optimized agent:")
for question in test_questions:
    test_response = optimized_agent.predict([
        ChatAgentMessage(role="user", content=question, id=str(uuid.uuid4()))
    ])
    print(f"Q: {question}")
    print(f"A: {test_response.messages[0].content}")
    print("-" * 50)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Log and Deploy the Optimized Agent

# COMMAND ----------
input_example = {"messages": [{"role": "user", "content": "Who is Zeus?"}]}

with mlflow.start_run(run_name="optimized_dspy_rag_agent_deployment"):
    
    # Prepare artifacts
    artifacts = {}
    if optimized_program_path:
        artifacts["optimized_program"] = optimized_program_path
    
    # Log the model with configuration and artifacts
    model_info = mlflow.pyfunc.log_model(
        name=model_name,
        python_model="agent.py",
        model_config=model_config.to_dict(),  # Pass the ModelConfig to the logged model
        artifacts=artifacts,
        pip_requirements=[
            f"dspy=={dspy.__version__}",
            f"databricks-agents=={get_distribution('databricks-agents').version}",
            f"mlflow=={mlflow.__version__}",
            "openai<2",
            f"databricks-sdk=={get_distribution('databricks-sdk').version}"
        ],
        resources=[
            DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT.replace("databricks/", "")),
            DatabricksVectorSearchIndex(index_name=UC_VS_INDEX_NAME),
            # DatabricksFunction(function_name=uc_function_name) # TODO: add function name if you have one
        ],
        input_example=input_example,
        code_paths=["agent.py", "utils.py"]
    )
    
    # Log additional metadata
    mlflow.log_param("uses_optimized_program", optimized_program_path is not None)
    mlflow.log_param("optimization_method", "MIPROv2")
    mlflow.log_param("config_file", config_path)
    mlflow.log_param("llm_endpoint", llm_config.get("endpoint"))
    mlflow.log_param("max_tokens", llm_config.get("max_tokens"))
    mlflow.log_param("temperature", llm_config.get("temperature"))
    vector_search_config = model_config.get("vector_search") or {}
    mlflow.log_param("top_k", vector_search_config.get("top_k"))
    
    # Log the model config as an artifact
    mlflow.log_dict(model_config.to_dict(), "deployment_config.json")
    
    print("Logged optimized model to:", model_info.model_uri)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the mlflow.models.predict() API. See Databricks documentation (AWS | Azure).
# COMMAND ----------
mlflow.models.predict(
    model_uri=f"runs:/{model_info.run_id}/{model_name}",
    input_data={"messages": [{"role": "user", "content": "Who is Zeus?"}]},
    env_manager="uv",
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register and Deploy

# COMMAND ----------
# Register the model
mlflow.set_registry_uri("databricks-uc")
uc_registered_model_info = mlflow.register_model(
    model_uri=model_info.model_uri, 
    name=UC_MODEL_NAME
)

# Deploy to Model Serving with configuration-based environment variables
from databricks import agents

# Build deployment environment variables from config
deployment_env_vars = {
    "DSPY_LLM_ENDPOINT": llm_config.get("endpoint", "databricks/databricks-claude-3-7-sonnet"),
    "VS_INDEX_FULLNAME": UC_VS_INDEX_NAME
}

# Add optimized program path if available
if optimized_program_path:
    deployment_env_vars["DSPY_OPTIMIZED_PROGRAM_PATH"] = optimized_program_path

# Deploy with configuration-based settings
agents.deploy(
    UC_MODEL_NAME, 
    uc_registered_model_info.version, 
    scale_to_zero=True, 
    environment_vars=deployment_env_vars,
    tags={
        "endpointSource": "dspy-rag-agent-optimized-v1",
        "configUsed": config_path,
        "optimized": str(optimized_program_path is not None)
    }
)

print(f"Deployed optimized agent: {UC_MODEL_NAME}")
print(f"Configuration: {config_path}")
print(f"Optimized program: {'Yes' if optimized_program_path else 'No'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC * Test the deployed agent in AI Playground
# MAGIC * Run evaluation using Mosaic AI Agent Evaluation
# MAGIC * Monitor performance in production
# MAGIC * Use the optimized configuration for better performance and cost efficiency