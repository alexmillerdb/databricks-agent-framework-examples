# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Optimized DSPy RAG Agent
# MAGIC 
# MAGIC This notebook deploys the optimized DSPy RAG agent to Databricks Model Serving.

# COMMAND ----------
%pip install -qqq --upgrade "mlflow[databricks]>=3.1" dspy-ai databricks-agents openai uv
dbutils.library.restartPython()

# COMMAND ----------
import uuid, os, mlflow, dspy
from pkg_resources import get_distribution
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatAgentChunk

# Configuration
catalog = "users"
schema = "alex_miller"
model_name = "dspy_rag_agent_optimized"
vector_search_index_name = "wikipedia_chunks_index"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"
UC_VS_INDEX_NAME = f"{catalog}.{schema}.{vector_search_index_name}"

# Set environment variables
os.environ["VS_INDEX_FULLNAME"] = UC_VS_INDEX_NAME

# MLflow configuration
mlflow.set_tracking_uri("databricks")
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f"/Users/{user_name}/dspy_rag_optimized_deployment")

# Enable DSPy tracing
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
    experiment = mlflow.get_experiment_by_name(f"/Users/{user_name}/dspy_rag_optimization")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No completed compilation runs found")
    
    latest_run = runs[0]
    
    # Download the optimized program
    artifact_path = client.download_artifacts(
        latest_run.info.run_id, 
        "optimized_program/optimized_rag_program.json"
    )
    
    # Set the path for the agent to use
    os.environ["DSPY_OPTIMIZED_PROGRAM_PATH"] = artifact_path
    
    print(f"Downloaded optimized program from run: {latest_run.info.run_id}")
    return artifact_path

# COMMAND ----------
# Download the optimized program
optimized_program_path = download_optimized_program()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test the Optimized Agent

# COMMAND ----------
from agent import DSPyRAGChatAgent, LLM_ENDPOINT

# Create agent with optimized program
optimized_agent = DSPyRAGChatAgent(use_optimized=True)

# Test the agent
test_response = optimized_agent.predict([
    ChatAgentMessage(role="user", content="Who is Zeus?", id=str(uuid.uuid4()))
])
print("Optimized Agent Response:", test_response.messages[0].content)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Log and Deploy the Optimized Agent

# COMMAND ----------
input_example = {"messages": [{"role": "user", "content": "Who is Zeus?"}]}

with mlflow.start_run(run_name="optimized_dspy_rag_agent_deployment"):
    
    # Log the model with the optimized program as an artifact
    model_info = mlflow.pyfunc.log_model(
        name="dspy_rag_agent_optimized",
        python_model="agent.py",
        artifacts={"optimized_program": optimized_program_path},
        # conda_env=conda_env,
        resources=[
            DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT.replace("databricks/", "")),
            DatabricksVectorSearchIndex(index_name=UC_VS_INDEX_NAME),
            # DatabricksFunction(function_name=uc_function_name) # TODO: add function name if you have one
        ]
    )
    
    # Log additional metadata
    mlflow.log_param("uses_optimized_program", True)
    mlflow.log_param("optimization_method", "MIPROv2")
    
    print("Logged optimized model to:", model_info.model_uri)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the mlflow.models.predict() API. See Databricks documentation (AWS | Azure).
# COMMAND ----------
mlflow.models.predict(
    model_uri=f"runs:/{model_info.run_id}/dspy_rag_agent_optimized",
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

# Deploy to Model Serving
from databricks import agents

agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, 
              scale_to_zero=True, 
              environment_vars={
                "DSPY_LLM_ENDPOINT": LLM_ENDPOINT,
                "VS_INDEX_FULLNAME": UC_VS_INDEX_NAME},
              tags={"endpointSource": "dspy-rag-agent-optimized-v1"})

print(f"Deployed optimized agent: {UC_MODEL_NAME}")