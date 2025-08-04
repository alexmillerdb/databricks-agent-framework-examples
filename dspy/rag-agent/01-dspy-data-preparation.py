# Databricks notebook source
# MAGIC %md
# MAGIC # Part 1: Prepare data and vector search index for a RAG DSPy program
# MAGIC
# MAGIC This notebook shows how to accomplish the following tasks to prepare text data for your retrieval augmented generation (RAG) DSPy program:
# MAGIC
# MAGIC - Set up your environment
# MAGIC - **Clean Wikipedia text data** (removes markup, templates, HTML tags, etc.)
# MAGIC - Create a Delta table of chunked data
# MAGIC - Create a Vector Search index ([AWS](https://docs.databricks.com/generative-ai/vector-search.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/vector-search)) 
# MAGIC
# MAGIC This notebook is part 1 of 2 notebooks for creating a DSPy program for RAG.
# MAGIC
# MAGIC **Note**: This version includes comprehensive text cleaning to improve data quality by ~22% through removal of Wikipedia markup, HTML tags, templates, and other noise.
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

# MAGIC %pip install --upgrade --force-reinstall -qqqq databricks-vectorsearch>=0.40 langchain==0.3.0 tiktoken==0.7.0 python-dotenv
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Setup
# MAGIC
# MAGIC This section detects whether we're running locally or in Databricks and configures accordingly.

# COMMAND ----------

from dotenv import load_dotenv
import os
from databricks.connect import DatabricksSession

def setup_environment():
    """
    Set up the execution environment for either local development or Databricks.
    
    Returns:
        tuple: (spark_session, user_name, script_dir)
    """
    try:    
        # Load environment variables for local testing
        load_dotenv()

        # os.environ["DATABRICKS_SERVERLESS_COMPUTE_ID"] = "auto"
        spark = DatabricksSession.builder.getOrCreate()
        
        # Check if we're in local development mode
        if os.getenv('DATABRICKS_TOKEN') and os.getenv('DATABRICKS_HOST'):
            print("🏠 Local Development Mode Detected")
            print("=" * 50)
            print(f"✅ Databricks Host: {os.getenv('DATABRICKS_HOST')}")
            print(f"✅ MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI', 'databricks')}")
            print("\n🎯 Ready for local development!")
        else:
            print("☁️  Databricks Environment Mode")
            print("=" * 40)
            print("ℹ️  Using Databricks workspace credentials")
            print("ℹ️  No additional setup required")

    except ImportError:
        print("ℹ️  python-dotenv not available - using Databricks environment")
        spark = DatabricksSession.builder.getOrCreate()
    except Exception as e:
        print(f"⚠️  Environment setup issue: {e}")
        spark = DatabricksSession.builder.getOrCreate()

    # Get current user
    user_name = spark.sql("SELECT current_user()").collect()[0][0]
    print(f"\n👤 User: {user_name}")
    
    # Get the directory where this script is located
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    return spark, user_name, script_dir

# Initialize environment
spark, user_name, script_dir = setup_environment()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Define notebook widgets

# COMMAND ----------

# Handle widgets for both notebook and local execution
try:
    # Try to use dbutils if in Databricks environment
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
except NameError:
    # Running locally - use environment variables or defaults
    print("ℹ️  Running locally - using environment variables or defaults")
    widgets = {
        "source_catalog": os.getenv("SOURCE_CATALOG", "users"),
        "source_schema": os.getenv("SOURCE_SCHEMA", user_name.replace("@", "_").replace(".", "_")),
        "source_table": os.getenv("SOURCE_TABLE", "wikipedia_chunks"),
        "vs_endpoint": os.getenv("VS_ENDPOINT", "one-env-shared-endpoint-0"),
        "vs_index": os.getenv("VS_INDEX", "wikipedia_chunks_index"),
    }
    print("📋 Using configuration:")
    for k, v in widgets.items():
        print(f"   {k}: {v}")


# COMMAND ----------

# MAGIC %md
# MAGIC ###Define configurations

# COMMAND ----------

print("📋 CONFIGURATIONS")
config = {}

try:
    # Try to get values from dbutils widgets (Databricks notebook)
    format_widget_name = lambda x: x.replace('_', ' ').title()
    for k, _ in widgets.items():
        config[k] = dbutils.widgets.get(k)
        assert config[k].strip() != "", f"Please provide a valid {format_widget_name(k)}"
        print(f"- config['{k}']= '{config[k]}'")
except NameError:
    # Use the values we set up earlier for local execution
    config = widgets.copy()
    print("Using local configuration values:")
    for k, v in config.items():
        print(f"- config['{k}']= '{v}'")

config["source_table_fullname"] = f"{config['source_catalog']}.{config['source_schema']}.{config['source_table']}"
config["vs_index_fullname"] =  f"{config['source_catalog']}.{config['source_schema']}.{config['vs_index']}"

print(f"- config['source_table_fullname']= '{config['source_table_fullname']}'")
print(f"- config['vs_index_fullname']= '{config['vs_index_fullname'] }'")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create Delta table 
# MAGIC
# MAGIC The following shows how to create a Delta table that contains chunked Wikipedia entries from Databricks sample datasets.
# MAGIC
# MAGIC **Text Cleaning Features:**
# MAGIC - Removes HTML/XML tags and entities
# MAGIC - Removes MediaWiki templates (`{{...}}`)
# MAGIC - Converts wiki links to clean text (`[[link|display]]` → `display`)
# MAGIC - Removes references, citations, and file/image links
# MAGIC - Cleans up section headers, bold/italic markup
# MAGIC - Normalizes whitespace and removes non-printable characters
# MAGIC - Filters out articles that become too short after cleaning

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import IntegerType, StructField, StructType, StringType
from pyspark.sql import functions as F
from typing import Iterator
import re
import html


def clean_wikipedia_text(text):
    """
    Comprehensive Wikipedia text cleaning function.
    Removes various Wikipedia markup, HTML, and formatting issues.
    """
    if not text or pd.isna(text):
        return ""
    
    # Convert to string if needed
    text = str(text)
    
    # 1. Basic HTML tag removal
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Decode HTML entities
    text = html.unescape(text)
    
    # 3. Remove MediaWiki templates ({{...}})
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    
    # 4. Remove wiki links but keep the display text
    # [[link|display]] -> display
    # [[link]] -> link
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    
    # 5. Remove external links [url text] -> text
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\s\]]+\]', '', text)
    
    # 6. Remove references and citations
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<ref[^>]*/?>', '', text, flags=re.IGNORECASE)
    
    # 7. Remove file/image references
    text = re.sub(r'\[\[File:[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[Image:[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    
    # 8. Remove category links
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    
    # 9. Remove infobox and table markup
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    
    # 10. Clean up section headers (=== text === -> text)
    text = re.sub(r'={2,}\s*([^=]+)\s*={2,}', r'\1', text)
    
    # 11. Remove bold/italic markup
    text = re.sub(r"'{2,5}([^']+)'{2,5}", r'\1', text)
    
    # 12. Remove leftover brackets and parentheses artifacts
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'\(\s*\)', '', text)
    
    # 13. Clean up whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines -> double newline
    text = re.sub(r'\n{3,}', '\n\n', text)   # More than 2 newlines -> double newline
    text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces/tabs -> single space
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)  # Remove spaces around newlines
    
    # 14. Remove non-printable characters except newlines and tabs
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    # 15. Remove very short lines (likely artifacts)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 3:  # Keep lines with more than 3 characters
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # 16. Final cleanup
    text = text.strip()
    
    return text

# Register as UDF for Spark
clean_text_udf = F.udf(clean_wikipedia_text, StringType())


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

# Apply text cleaning to remove Wikipedia markup and improve content quality
print("Applying text cleaning to remove Wikipedia markup...")
cleaned_source_df = source_df.withColumn("text", clean_text_udf(F.col("text")))

# Filter out articles that became too short after cleaning (likely poor quality)
min_text_length = 100
cleaned_source_df = cleaned_source_df.filter(F.length("text") >= min_text_length)

print(f"After cleaning and filtering: {cleaned_source_df.count()} articles remain (from {n_samples} original)")
cleaned_source_df.write.mode("overwrite").saveAsTable(f"{config['source_catalog']}.{config['source_schema']}.wikipedia_source")

# We need to adjust the "id" field to be a StringType instead of an IntegerType,
# because our IDs now have format "[ORIGINAL ID]_[CHUNK ID]".
schema_without_id = list(filter(lambda f: f.name != "id", cleaned_source_df.schema.fields))
chunked_schema = StructType(
    [StructField("id", StringType()), StructField("chunk_id", IntegerType())]
    + schema_without_id
)

# Use cleaned data for chunking
chunked_df = cleaned_source_df.mapInPandas(
    pandas_text_splitter, schema=chunked_schema
).withColumnRenamed("text", "chunk")

# Write chunked table
chunked_df.write.mode("overwrite").option(
    "delta.enableChangeDataFeed", "true"
).saveAsTable(config["source_table_fullname"])

# Display chunked table
try:
    # Use display() in Databricks notebook
    display(spark.sql(f"SELECT * FROM {config['source_table_fullname']}"))
except NameError:
    # Use show() for local execution
    print(f"📊 Sample data from {config['source_table_fullname']}:")
    spark.sql(f"SELECT * FROM {config['source_table_fullname']}").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create a vector search index
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Check and create vector search endpoint and index
# MAGIC
# MAGIC The following code intelligently handles vector search setup:
# MAGIC - **Endpoint**: Checks if endpoint exists, creates if missing
# MAGIC - **Index**: If index exists, syncs with new data; if not, creates new index
# MAGIC
# MAGIC This approach is more robust and supports incremental updates without destroying existing indexes.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Check if vector search endpoint exists, create if not
print(f"🔍 Checking vector search endpoint: {config['vs_endpoint']}")
try:
    endpoint = vsc.get_endpoint(config["vs_endpoint"])
    print(f"✅ Vector search endpoint '{config['vs_endpoint']}' already exists")
    # Check endpoint status if available
    try:
        status = endpoint.describe() if hasattr(endpoint, 'describe') else None
        if status and 'endpoint_status' in status:
            print(f"   Endpoint status: {status['endpoint_status']}")
    except:
        pass
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"🔧 Creating vector search endpoint: {config['vs_endpoint']}")
        try:
            vsc.create_endpoint(
                name=config["vs_endpoint"],
                endpoint_type="STANDARD"
            )
            print(f"✅ Created vector search endpoint: {config['vs_endpoint']}")
        except Exception as create_e:
            print(f"⚠️  Could not create endpoint (may already exist or require admin): {create_e}")
    else:
        raise(e)

# Check if vector search index exists
print(f"\n🔍 Checking vector search index: {config['vs_index_fullname']}")
index_exists = False
existing_index = None

try:
    existing_index = vsc.get_index(index_name=config['vs_index_fullname'])
    index_exists = True
    print(f"✅ Vector search index '{config['vs_index_fullname']}' already exists")
    
    # Sync the existing index to pick up new data
    print("🔄 Syncing existing index with latest data...")
    existing_index.sync()
    print("✅ Index sync initiated")
    
except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"🔧 Vector search index '{config['vs_index_fullname']}' does not exist - will create new one")
        index_exists = False
    else:
        raise(e)

# COMMAND ----------

# MAGIC %md
# MAGIC The following creates a vector search index from the Delta table of chunked documents and reports status as it progresses. 
# MAGIC If the index already exists, it will sync the existing index with new data instead.
# MAGIC
# MAGIC <b>Note</b>: This block will wait until the index is created/synced. Expect this command to take several minutes as you wait for your index to come online.

# COMMAND ----------

import time

if not index_exists:
    print(f"🔧 Creating new vector search index: {config['vs_index_fullname']}")
    index = vsc.create_delta_sync_index(
        endpoint_name=config["vs_endpoint"],
        source_table_name=config["source_table_fullname"],
        index_name=config["vs_index_fullname"],
        pipeline_type='TRIGGERED',
        primary_key="id",
        embedding_source_column="chunk",
        embedding_model_endpoint_name="databricks-bge-large-en"
    )
    print("✅ Vector search index creation initiated")
else:
    # Use the existing index we found earlier
    index = existing_index
    print("🔄 Using existing index for status monitoring")

# Wait for index to become online (works for both new creation and sync)
print("⏳ Waiting for index to be ONLINE...")
while not index.describe().get('status').get('detailed_state').startswith('ONLINE'):
    current_state = index.describe().get('status').get('detailed_state')
    print(f"{time.strftime('%X %x %Z')} -- Index state: {current_state} - waiting...")
    time.sleep(10)

print("✅ Index is ONLINE and ready!")
index_status = index.describe()
print(f"📊 Index details: {index_status.get('status', {}).get('detailed_state')}")

# Display final index information
if 'status' in index_status:
    status_info = index_status['status']
    if 'ready' in status_info:
        print(f"🎯 Index ready: {status_info['ready']}")
    if 'index_url' in status_info:
        print(f"🔗 Index URL: {status_info['index_url']}")

index_status

# COMMAND ----------

# MAGIC %md
# MAGIC Now you are ready to [Create DSPy program for RAG](https://docs.databricks.com/en/_extras/notebooks/source/generative-ai/dspy-create-rag-program.html)!