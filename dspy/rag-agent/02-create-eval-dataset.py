# Databricks notebook source
# MAGIC %md ## MLflow Evaluation Dataset Creation Notebook

# COMMAND ----------

# MAGIC %pip install databricks-agents "mlflow[databricks]>=3.1.0"
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import mlflow.genai.datasets
from pyspark.sql import functions as F
from databricks.connect import DatabricksSession
from databricks.agents.evals import generate_evals_df
import pandas as pd
import math

catalog = "users"
schema = "alex_miller"
table = "wikipedia_source"
evaluation_table = "dspy_rag_eval_example"

# load MLflow config
config_path = "config.yaml"
model_config = mlflow.models.ModelConfig(development_config=config_path)
mlflow_config = model_config.get("mlflow_config") or {}

# Ensure we talk to the Databricks tracking server for run + model logging
mlflow.set_tracking_uri("databricks")

# Set experiment based on config
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = mlflow_config.get("experiment_name", "dspy_rag_chat_agent")
experiment_info = mlflow.set_experiment(f"/Users/{user_name}/{experiment_name}")
print(f"MLflow Experiment info: {experiment_info}")

# COMMAND ----------

try:
    spark = DatabricksSession.builder.remote(serverless=True).getOrCreate()
except Exception as e:
    pass

docs = (spark.read.table(f"{catalog}.{schema}.{table}")
        .select(F.concat(F.col("title").alias("content"), F.lit(": "), F.col("text")).alias("content"),
                F.monotonically_increasing_id().cast("string").alias("doc_uri"))
        .limit(100)
        )

display(docs)

agent_description = """
The Agent is a RAG chatbot that answers questions about Wikipedia documents and articles. 
The Agent has access to a corpus of Wikipedia documents and articles, and its task is to answer the user's questions by retrieving the 
relevant docs from the corpus and synthesizing a helpful, accurate response.
"""

question_guidelines = """
# Example questions
- Who is Hercules dad?

# Additional Guidelines
- Questions should be succinct, and human-like
"""

num_evals = 100

evals = generate_evals_df(
    docs,
    # The total number of evals to generate. The method attempts to generate evals that have full coverage over the documents
    # provided. If this number is less than the number of documents, is less than the number of documents,
    # some documents will not have any evaluations generated. See "How num_evals is used" below for more details.
    num_evals=num_evals,
    # A set of guidelines that help guide the synthetic generation. These are free-form strings that will be used to prompt the generation.
    agent_description=agent_description,
    question_guidelines=question_guidelines
)

display(evals)

# create a table in the users.alex_miller schema
spark_evals = spark.createDataFrame(evals)
spark_evals.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.wikipedia_synthetic_eval")

# COMMAND ----------

evaluation_uc_table_name = f"{catalog}.{schema}.{evaluation_table}"

# Load or create MLflow evaluation dataset
try:
    dataset = mlflow.genai.datasets.get_dataset(
        uc_table_name=evaluation_uc_table_name
    )
except Exception as e:
    print(f"MLflow evaluation dataset {evaluation_uc_table_name} not found... Creating dataset")
    dataset = mlflow.genai.datasets.create_dataset(
        uc_table_name=evaluation_uc_table_name,
        experiment_id=experiment_info.experiment_id
    )