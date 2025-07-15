# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1: Prepare data and vector search index for a RAG DSPy program
# MAGIC
# MAGIC This notebook shows how to accomplish the following tasks to prepare text data for your retrieval augmented generation (RAG) DSPy program:
# MAGIC
# MAGIC - Set up your environment
# MAGIC - Create a Delta table of chunked data
# MAGIC - Create a Vector Search index ([AWS](https://docs.databricks.com/generative-ai/vector-search.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/vector-search)) 
# MAGIC
# MAGIC This notebook is part 1 of 2 notebooks for creating a DSPy program for RAG.
# MAGIC
# MAGIC ## Requirement
# MAGIC
# MAGIC - Create a vector search endpoint ([AWS](https://docs.databricks.com/generative-ai/create-query-vector-search.html#create-a-vector-search-endpoint) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/create-query-vector-search#create-a-vector-search-endpoint))
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Install dependencies

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall -qqqq databricks-vectorsearch>=0.40 langchain==0.3.0 tiktoken==0.7.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Define notebook widgets

# COMMAND ----------

dbutils.widgets.removeAll()
format_widget_name = lambda x: x.replace('_', ' ').title()

widgets = {
    "source_catalog": "users",
    "source_schema": "alex_miller",
    "source_table": "wikipedia_chunks",
    "vs_endpoint": "one-env-shared-endpoint-0",
    "vs_index": "wikipedia_chunks_index",
}
for k, v in widgets.items():
    dbutils.widgets.text(k, v, format_widget_name(k))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Define configurations

# COMMAND ----------

print("CONFIGURATIONS")
config = {}
for k, _ in widgets.items():
    config[k] = dbutils.widgets.get(k)
    assert config[k].strip() != "", f"Please provide a valid {format_widget_name(k)}"
    print(f"- config['{k}']= '{config[k]}'")

config["source_table_fullname"] = f"{config['source_catalog']}.{config['source_schema']}.{config['source_table']}"
config["vs_index_fullname"] =  f"{config['source_catalog']}.{config['source_schema']}.{config['vs_index']}"

print(f"- config['source_table_fullname']= '{config['source_table_fullname']}'")
print(f"- config['vs_index_fullname']= '{config['vs_index_fullname'] }'")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create Delta table 
# MAGIC
# MAGIC The following shows how to create a Delta table that contains chunked Wikipedia entries from Databricks sample datasets.

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType, StructField, StructType, StringType
from typing import Iterator


# Define a pandas UDF to chunk text
def pandas_text_splitter(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Apache Spark pandas UDF for chunking text in a scale-out pipeline.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    for pdf in iterator:
        pdf["text"] = pdf["text"].apply(text_splitter.split_text)
        chunk_pdf = pdf.explode("text")
        chunk_pdf_with_index = chunk_pdf.reset_index().rename(
            columns={"index": "chunk_id"}
        )
        chunk_ids = chunk_pdf_with_index.groupby("chunk_id").cumcount()
        # Define ids with format "[ORIGINAL ID]_[CHUNK ID]"
        chunk_pdf_with_index["id"] = (
            chunk_pdf_with_index["id"].astype("str") + "_" + chunk_ids.astype("str")
        )
        yield chunk_pdf_with_index


# Accessing Wikipedia sample data
source_data_path = "dbfs:/databricks-datasets/wikipedia-datasets/data-001/en_wikipedia/articles-only-parquet"

n_samples = 5000
source_df = spark.read.parquet(source_data_path).limit(n_samples)

# We need to adjust the "id" field to be a StringType instead of an IntegerType,
# because our IDs now have format "[ORIGINAL ID]_[CHUNK ID]".
schema_without_id = list(filter(lambda f: f.name != "id", source_df.schema.fields))
chunked_schema = StructType(
    [StructField("id", StringType()), StructField("chunk_id", IntegerType())]
    + schema_without_id
)

chunked_df = source_df.mapInPandas(
    pandas_text_splitter, schema=chunked_schema
).withColumnRenamed("text", "chunk")

# Write chunked table
chunked_df.write.mode("overwrite").option(
    "delta.enableChangeDataFeed", "true"
).saveAsTable(config["source_table_fullname"])

# dislpay chunked table
display(spark.sql(f"SELECT * FROM {config['source_table_fullname']}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a vector search index
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC The following deletes any previously existing vector search index with the same name. You can skip this command if you don't have a vector search index with that name.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

try:  
  vsc.delete_index(index_name=config['vs_index_fullname'])
except Exception as e:
  if "RESOURCE_DOES_NOT_EXIST" in str(e):
    print(f"\n\tThe VS index: {config['vs_index_fullname']} does not exist")
  else:  
    raise(e)

# COMMAND ----------

# MAGIC %md
# MAGIC The following creates a vector search index from the Delta table of chunked documents and reports status as it progresses. 
# MAGIC
# MAGIC <b>Note</b>: This block will wait until the index is created. Expect this command to take several minutes as you wait for your index to come online.

# COMMAND ----------

import time

index = vsc.create_delta_sync_index(
  endpoint_name=config["vs_endpoint"],
  source_table_name=config["source_table_fullname"],
  index_name=config["vs_index_fullname"],
  pipeline_type='TRIGGERED',
  primary_key="id",
  embedding_source_column="chunk",
  embedding_model_endpoint_name="databricks-bge-large-en"
)

# Wait for index to become online. Expect this command to take several minutes.
while not index.describe().get('status').get('detailed_state').startswith('ONLINE'):
  print(f"{time.strftime('%X %x %Z')} -- Waiting for index to be ONLINE...  ")
  time.sleep(5)
print("Index is ONLINE")
index.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Now you are ready to [Create DSPy program for RAG](https://docs.databricks.com/en/_extras/notebooks/source/generative-ai/dspy-create-rag-program.html)!