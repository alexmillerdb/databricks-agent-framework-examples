# Databricks notebook source
# MAGIC %md
# MAGIC # DSPy RAG ➡️ MLflow ChatAgent (PyFunc)
# MAGIC 
# MAGIC This notebook-style script shows how to:
# MAGIC 
# MAGIC * Wrap that DSPy program with the **MLflow `ChatAgent`** interface so it can be served like any other Databricks Agent.
# MAGIC * Enable **MLflow 3.x tracing** for full DSPy span capture.
# MAGIC * Log the agent as an **MLflow PyFunc model** that can be deployed to Model Serving or used in the AI Playground.
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
# MAGIC ## Imports & MLflow / DSPy configuration

# COMMAND ----------
import uuid, os, mlflow, dspy
from pkg_resources import get_distribution
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex, DatabricksFunction
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatAgentChunk

# Enable full DSPy autologging (captures compile/eval + live inference traces)
mlflow.dspy.autolog()

# Ensure we talk to the Databricks tracking server for run + model logging
mlflow.set_tracking_uri("databricks")

# If you want to store experiments in a dedicated path, change this value
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_info = mlflow.set_experiment(f"/Users/{user_name}/dspy_rag_chat_agent")
print(f"MLflow Experiment info: {experiment_info}")

# Set the LLM endpoint name (defaults to "databricks/databricks-claude-3-7-sonnet")
os.environ["DSPY_LLM_ENDPOINT"] = "databricks/databricks-claude-3-7-sonnet"

# Set the VS index name created in previous step
os.environ["VS_INDEX_FULLNAME"] = UC_VS_INDEX_NAME

# -----------------------------------------------------------------------------
# Import the ChatAgent class directly from `agent.py`
# -----------------------------------------------------------------------------
from agent import DSPyRAGChatAgent, LLM_ENDPOINT  # type: ignore

# Instantiate the agent for logging
chat_agent = DSPyRAGChatAgent()
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

with mlflow.start_run(run_name="dspy_rag_chat_agent_build"):
    model_info = mlflow.pyfunc.log_model(
        name="dspy_rag_agent",
        python_model="agent.py",
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
        ]
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
    model_uri=f"runs:/{model_info.run_id}/dspy_rag_agent",
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
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, 
              scale_to_zero=True, 
              environment_vars={
                "DSPY_LLM_ENDPOINT": LLM_ENDPOINT,
                "VS_INDEX_FULLNAME": UC_VS_INDEX_NAME},
              tags={"endpointSource": "dspy-rag-agent-v1"})
# COMMAND ----------
# MAGIC %md
# MAGIC ### Next steps
# MAGIC * Deploy to Model Serving for scalable, guarded inference.
# MAGIC * Open the run’s *Tracing* tab in the MLflow UI to inspect DSPy spans.
# MAGIC * Evaluate with Mosaic AI Agent Evaluation notebooks. 