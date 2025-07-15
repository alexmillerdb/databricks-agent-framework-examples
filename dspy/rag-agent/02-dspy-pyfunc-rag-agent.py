# Databricks notebook source
# MAGIC %md
# MAGIC # DSPy RAG ➡️ MLflow ChatAgent (PyFunc)
# MAGIC 
# MAGIC This notebook-style script shows how to:
# MAGIC 
# MAGIC * Wrap that DSPy program with the **MLflow `ChatAgent`** interface so it can be served like any other Databricks Agent.
# MAGIC * Enable **MLflow 3.x tracing** for full DSPy span capture.
# MAGIC * Log the agent as an **MLflow PyFunc model** that can be deployed to Model Serving or used in the AI Playground.
# MAGIC * Use **MLflow ModelConfig** for parameterized deployment across environments.
# MAGIC 
# MAGIC The implementation below merges the ideas from the official Databricks notebooks
# MAGIC * *dspy-create-rag-program.py* (building the RAG module)
# MAGIC * *dspy-pyfunc-simple-agent* (wrapping in a `ChatAgent`)
# MAGIC * *MLflow 3.0 ↔︎ DSPy tracing tutorial*
# MAGIC 
# MAGIC **Prerequisites**
# MAGIC * Databricks Runtime 15.4+ (or DBR 15.x) with Python 3.11+ (Serverless CPU version 3)
# MAGIC * `mlflow>=3.1`, `dspy-ai>=2.5`, `databricks-agents>=0.5`
# MAGIC * A Vector Search index prepared as described in Part 1 of the tutorial.
# MAGIC * A Databricks FM API Model endpoint (or pay-per-token LLM) accessible via `dspy.LM`. 

# COMMAND ----------
# MAGIC %md
# MAGIC ## Install (or upgrade) required packages
# MAGIC Running `%pip install` in a Databricks notebook cell automatically triggers a Python restart.

# COMMAND ----------
%pip install -qqq --upgrade "mlflow[databricks]>=3.1" dspy-ai databricks-agents openai uv
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Define the catalog, schema, vector search index and model name
# COMMAND ----------

catalog = "users"
schema = "alex_miller"
model_name = "dspy_rag_agent"
vector_search_index_name = "wikipedia_chunks_index"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"
UC_VS_INDEX_NAME = f"{catalog}.{schema}.{vector_search_index_name}"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration Setup
# MAGIC 
# MAGIC Load configuration using MLflow ModelConfig for parameterized deployment.

# COMMAND ----------
import mlflow
import os

# Load development configuration
config_path = "config.yaml"
model_config = mlflow.models.ModelConfig(development_config=config_path)

# Set environment variables from config
os.environ["VS_INDEX_FULLNAME"] = UC_VS_INDEX_NAME
llm_config = model_config.get("llm_config") or {}
os.environ["DSPY_LLM_ENDPOINT"] = llm_config.get("endpoint", "databricks/databricks-claude-3-7-sonnet")

print(f"Using LLM endpoint: {os.environ['DSPY_LLM_ENDPOINT']}")
print(f"Using Vector Search index: {UC_VS_INDEX_NAME}")
print(f"Configuration loaded from: {config_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Imports & MLflow / DSPy configuration

# COMMAND ----------
import uuid, dspy
from pkg_resources import get_distribution
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex, DatabricksFunction
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatAgentChunk

# Configure MLflow based on config
mlflow_config = model_config.get("mlflow_config") or {}
if mlflow_config.get("enable_autolog", True):
    # Enable full DSPy autologging (captures compile/eval + live inference traces)
    mlflow.dspy.autolog()

# Ensure we talk to the Databricks tracking server for run + model logging
mlflow.set_tracking_uri("databricks")

# Set experiment based on config
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = mlflow_config.get("experiment_name", "dspy_rag_chat_agent")
experiment_info = mlflow.set_experiment(f"/Users/{user_name}/{experiment_name}")
print(f"MLflow Experiment info: {experiment_info}")

# -----------------------------------------------------------------------------
# Import the ChatAgent class directly from `agent.py`
# -----------------------------------------------------------------------------
from agent import DSPyRAGChatAgent, LLM_ENDPOINT  # type: ignore

# Instantiate the agent with configuration
chat_agent = DSPyRAGChatAgent(config=model_config)
test_response = chat_agent.predict([
    ChatAgentMessage(role="user", content="Who is Zeus?", id=str(uuid.uuid4()))
])
print("Assistant:", test_response.messages[0].content)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Log the ChatAgent as an MLflow PyFunc model
# MAGIC This enables deployment to Model Serving or direct loading via `mlflow.pyfunc.load_model()`.

# COMMAND ----------
input_example = {"messages": [{"role": "user", "content": "Who is Zeus?"}]}

# Get model name from config
model_name_from_config = mlflow_config.get("registered_model_name", "dspy_rag_agent")

with mlflow.start_run(run_name="dspy_rag_chat_agent_build"):
    model_info = mlflow.pyfunc.log_model(
        name=model_name_from_config,
        python_model="agent.py",
        model_config=model_config.to_dict(),  # Pass the ModelConfig to the logged model
        # Pin package versions for reproducibility
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
    print("Logged model to:", model_info.model_uri)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load & use the logged model (optional local validation)

# COMMAND ----------
loaded_agent = mlflow.pyfunc.load_model(model_info.model_uri)

result = loaded_agent.predict(input_example)

print("Prediction via loaded agent:", result)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Pre-deployment agent validation
# MAGIC Before registering and deploying the agent, perform pre-deployment checks using the mlflow.models.predict() API. See Databricks documentation (AWS | Azure).
# COMMAND ----------
mlflow.models.predict(
    model_uri=f"runs:/{model_info.run_id}/{model_name_from_config}",
    input_data={"messages": [{"role": "user", "content": "Who is Zeus?"}]},
    env_manager="uv",
)
# COMMAND ----------
# MAGIC %md
# MAGIC ## Register the logged model to Unity Catalog
# COMMAND ----------
mlflow.set_registry_uri("databricks-uc")

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=model_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Deploy the model to Model Serving
# COMMAND ----------
from databricks import agents

# Get deployment configuration
deployment_llm_config = model_config.get("llm_config") or {}
deployment_config = {
    "DSPY_LLM_ENDPOINT": deployment_llm_config.get("endpoint", "databricks/databricks-claude-3-7-sonnet"),
    "VS_INDEX_FULLNAME": UC_VS_INDEX_NAME
}

agents.deploy(
    UC_MODEL_NAME, 
    uc_registered_model_info.version, 
    scale_to_zero=True, 
    environment_vars=deployment_config,
    tags={"endpointSource": "dspy-rag-agent-v1", "configUsed": config_path}
)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Next steps
# MAGIC * Deploy to Model Serving for scalable, guarded inference.
# MAGIC * Open the run's *Tracing* tab in the MLflow UI to inspect DSPy spans.
# MAGIC * Evaluate with Mosaic AI Agent Evaluation notebooks.
# MAGIC * Use `config_optimized.yaml` for production deployment with optimized settings. 