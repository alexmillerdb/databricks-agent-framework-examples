# Databricks notebook source
# MAGIC %md
# MAGIC # Build and Optimize DSPy RAG Agent
# MAGIC
# MAGIC This notebook provides a clean orchestration workflow for building, optimizing, and deploying DSPy-based RAG agents:
# MAGIC 1. **Environment Setup** - Configure for local or Databricks execution
# MAGIC 2. **Build Base Agent** - Create DSPy RAG agent with MLflow ChatAgent interface
# MAGIC 3. **Optimize Agent** (Optional) - Use DSPy compilation to improve performance
# MAGIC 4. **Log to MLflow** - Register the agent for deployment
# MAGIC 5. **Deploy** (Optional) - Deploy to Model Serving endpoint
# MAGIC
# MAGIC ## Requirements
# MAGIC - Databricks Vector Search index created (run `01-dspy-data-preparation.py`)
# MAGIC - Evaluation dataset (optional, run `02-create-eval-dataset.py` for optimization)
# MAGIC - Unity Catalog permissions for model registration
# MAGIC
# MAGIC ## Modular Architecture
# MAGIC This script now uses a modular architecture with the following modules:
# MAGIC - `optimizer.py` - DSPy optimization workflow and strategies
# MAGIC - `deploy.py` - MLflow logging, model registration, and deployment
# MAGIC - `utils.py` - General utilities and environment setup
# MAGIC - `agent.py` - MLflow ChatAgent implementation
# MAGIC - `metrics.py` - Evaluation metrics and scoring

"""
DSPy RAG Agent Builder and Optimizer - Modular Orchestration Script

A clean orchestration script for building, optimizing, and deploying DSPy-based RAG agents
using MLflow and Databricks. All complex logic has been extracted to dedicated modules
for better maintainability, testability, and reusability.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration and Setup
# MAGIC
# MAGIC This section sets up the workflow configuration and environment.

# COMMAND ----------

import os
from datetime import datetime

import mlflow
import dspy

# Import our modular components  
from modules.utils import (
    setup_environment, 
    print_workflow_configuration, 
    build_retriever,
    format_duration
)
from modules.optimizer import (
    run_optimization_workflow,
    get_optimization_config
)
from modules.deploy import (
    create_final_agent,
    full_deployment_workflow
)
from agent import DSPyRAGChatAgent, _DSPyRAGProgram

# COMMAND ----------

# MAGIC %md
# MAGIC ## Workflow Configuration
# MAGIC
# MAGIC Configure the workflow behavior. Set these variables to control what the script does.

# COMMAND ----------

# === WORKFLOW CONFIGURATION ===
# Main workflow flags
OPTIMIZE_AGENT = True              # Set to True to run DSPy optimization
DEPLOY_MODEL = True                # Set to True to deploy to Model Serving
USE_FAST_OPTIMIZATION = True       # Set to True for 5-10 minute optimization (testing)

# Configuration file  
CONFIG_FILE = "config/config.yaml"

# Unity Catalog settings
UC_CATALOG = "users"
UC_SCHEMA = "alex_miller" 
UC_MODEL_NAME = "dspy_rag_agent"

# Evaluation dataset (optional - set to None to use hardcoded examples)
EVAL_DATASET_NAME = "wikipedia_synthetic_eval"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Setup

# COMMAND ----------

# Initialize environment
print("🚀 Starting DSPy RAG Agent Builder")
print("=" * 50)

spark, user_name, script_dir = setup_environment()

# Load model configuration
model_config = mlflow.models.ModelConfig(development_config=os.path.join(script_dir, CONFIG_FILE))
config_dict = model_config.to_dict()

# Extract configuration sections
llm_config = config_dict.get("llm_config", {})
vector_search_config = config_dict.get("vector_search", {})

# Build dataset table reference
eval_dataset_table = f"{UC_CATALOG}.{UC_SCHEMA}.{EVAL_DATASET_NAME}" if EVAL_DATASET_NAME else None

# Get optimization configuration
optimization_config = get_optimization_config("fast" if USE_FAST_OPTIMIZATION else "production")

# Print workflow configuration
print_workflow_configuration(
    optimize_agent=OPTIMIZE_AGENT,
    deploy_model=DEPLOY_MODEL,
    config_file=CONFIG_FILE,
    eval_dataset_table=eval_dataset_table,
    optimization_config=optimization_config
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Setup

# COMMAND ----------

print("\n📊 Setting up MLflow...")

# Set MLflow registry to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Setup MLflow experiment
experiment_name = f"/Users/{user_name}/dspy_rag_agent_experiments"
mlflow.set_experiment(experiment_name)

print(f"✅ MLflow experiment: {experiment_name}")
print(f"✅ Registry URI: {mlflow.get_registry_uri()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## DSPy Language Model Configuration

# COMMAND ----------

print("\n🤖 Configuring DSPy Language Models...")

# Main LM for generation
endpoint = llm_config.get("endpoint", "databricks/databricks-claude-3-7-sonnet")
max_tokens = llm_config.get("max_tokens", 2500)
temperature = llm_config.get("temperature", 0.01)

print(f"🎯 Generation LM: {endpoint}")
print(f"   Max tokens: {max_tokens}, Temperature: {temperature}")

_lm = dspy.LM(
    endpoint,
    cache=False,
    max_tokens=max_tokens,
    temperature=temperature
)

# Configure separate LM for optimization evaluation judges
optimization_judge_config = config_dict.get("llm_endpoints", {}).get("optimization_judge", llm_config)
judge_endpoint = optimization_judge_config.get("endpoint", "databricks/databricks-claude-3-7-sonnet")

print(f"⚖️  Judge LM: {judge_endpoint}")

_judge_lm = dspy.LM(
    judge_endpoint,
    cache=False,
    max_tokens=optimization_judge_config.get("max_tokens", 1000),
    temperature=optimization_judge_config.get("temperature", 0.0)
)

# Enable DSPy autologging for optimization tracking
mlflow.dspy.autolog()

print("✅ DSPy language models configured")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Base Agent

# COMMAND ----------

print("\n🔧 Building base DSPy RAG agent...")

# Build retriever
retriever = build_retriever(config_dict)

# Create base DSPy program
base_program = _DSPyRAGProgram(retriever)

# Create base agent
base_agent = DSPyRAGChatAgent(rag_program=base_program, config=model_config)

print("✅ Base agent created successfully")

# Test base agent
print("\n🧪 Testing base agent...")
from mlflow.types.agent import ChatAgentMessage

test_messages = [ChatAgentMessage(role="user", content="Who is Zeus in Greek mythology?")]
test_response = base_agent.predict(messages=test_messages)
print(f"📤 Base agent response preview: {test_response.messages[0].content[:150]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimization Workflow (Optional)

# COMMAND ----------

optimized_program = None
optimized_program_path = None
optimization_results = None

if OPTIMIZE_AGENT:
    print("\n🚀 Starting optimization workflow...")
    start_time = datetime.now()
    
    try:
        optimized_program, optimized_program_path, optimization_results = run_optimization_workflow(
            spark=spark,
            lm=_lm,
            judge_lm=_judge_lm,
            base_program=base_program,
            config=optimization_config,
            uc_catalog=UC_CATALOG,
            uc_schema=UC_SCHEMA,
            script_dir=script_dir,
            eval_dataset_name=EVAL_DATASET_NAME,
            use_fast_config=USE_FAST_OPTIMIZATION
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 Optimization completed in {format_duration(duration)}")
        
        if optimization_results:
            print(f"📊 Optimization Results:")
            print(f"   Baseline: {optimization_results.get('baseline_score', 0):.3f}")
            print(f"   Final: {optimization_results.get('final_score', 0):.3f}")
            print(f"   Improvement: {optimization_results.get('improvement', 0):.3f} ({optimization_results.get('improvement_percent', 0):.1f}%)")
            print(f"   Strategy: {optimization_results.get('strategy')}")
            
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("⚠️  Continuing with base agent...")
        OPTIMIZE_AGENT = False  # Disable optimization flag for downstream logic
        
else:
    print("\n⏭️  Skipping optimization - using base agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Final Agent

# COMMAND ----------

print("\n📦 Creating final agent...")

# Create final agent with optimization if available
final_agent, final_config = create_final_agent(
    model_config=model_config,
    optimized_program=optimized_program,
    optimized_program_path=optimized_program_path,
    optimize_agent=OPTIMIZE_AGENT
)

print("✅ Final agent created successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Logging and Deployment

# COMMAND ----------

if DEPLOY_MODEL:
    print("\n🚀 Starting deployment workflow...")
    
    try:
        model_info, deployment = full_deployment_workflow(
            final_config=final_config,
            llm_config=llm_config,
            vector_search_config=vector_search_config,
            script_dir=script_dir,
            uc_catalog=UC_CATALOG,
            uc_schema=UC_SCHEMA,
            uc_model_name=UC_MODEL_NAME,
            config_file=CONFIG_FILE,
            optimized_program_path=optimized_program_path,
            optimization_results=optimization_results,
            optimize_agent=OPTIMIZE_AGENT,
            deploy_to_serving=True
        )
        
        print(f"\n🎉 Deployment completed successfully!")
        print(f"📍 Model URI: {model_info.model_uri}")
        print(f"📝 Model Name: {UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}")
        
        if deployment:
            print(f"🔗 Endpoint: {deployment.endpoint_name}")
            print(f"📊 Status: {deployment.state}")
            
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        print("ℹ️  Check Unity Catalog permissions and retry")
        
else:
    print("\n⏭️  Skipping endpoint deployment - logging and testing model only")
    
    # Log model and test it (but don't deploy to serving endpoint)
    from modules.deploy import log_model_to_mlflow, test_logged_model
    
    print("\n📋 Step 1: Logging model to MLflow...")
    model_info = log_model_to_mlflow(
        final_config=final_config,
        llm_config=llm_config,
        vector_search_config=vector_search_config,
        script_dir=script_dir,
        uc_model_name=UC_MODEL_NAME,
        config_file=CONFIG_FILE,
        optimized_program_path=optimized_program_path,
        optimization_results=optimization_results,
        optimize_agent=OPTIMIZE_AGENT
    )
    
    print("\n📋 Step 2: Testing logged model...")
    test_logged_model(model_info)
    print(f"✅ Model logged and tested successfully: {model_info.model_uri}")
    print("ℹ️  Set DEPLOY_MODEL=True to deploy to Model Serving endpoint")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Next Steps

# COMMAND ----------

print("\n" + "="*60)
print("🎉 DSPy RAG Agent Builder - Workflow Complete!")
print("="*60)

print(f"\n📋 Workflow Summary:")
print(f"   ✅ Base agent created")
print(f"   {'✅' if OPTIMIZE_AGENT and optimization_results else '⏭️ '} Optimization: {'Completed' if OPTIMIZE_AGENT and optimization_results else 'Skipped'}")
print(f"   ✅ Model logged to MLflow")
print(f"   {'✅' if 'deployment' in locals() and deployment else '⏭️ '} Deployment: {'Completed' if 'deployment' in locals() and deployment else 'Skipped'}")

if OPTIMIZE_AGENT and optimization_results:
    print(f"\n🏆 Performance Results:")
    baseline = optimization_results.get('baseline_score', 0)
    final = optimization_results.get('final_score', 0) 
    improvement = optimization_results.get('improvement_percent', 0)
    print(f"   📊 Baseline Score: {baseline:.3f}")
    print(f"   🚀 Final Score: {final:.3f}")
    print(f"   📈 Improvement: {improvement:.1f}%")
    print(f"   ⚡ Mode: {'Fast (5-10 min)' if USE_FAST_OPTIMIZATION else 'Production'}")

print(f"\n📁 Artifacts:")
print(f"   📍 Model: {UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}")
if 'model_info' in locals():
    print(f"   🔗 MLflow URI: {model_info.model_uri}")
if optimized_program_path:
    print(f"   🎯 Optimized Program: {optimized_program_path}")

print(f"\n🚀 Next Steps:")
if not OPTIMIZE_AGENT:
    print(f"   🔄 Run with OPTIMIZE_AGENT=True for better performance")
if not DEPLOY_MODEL:
    print(f"   🚀 Run with DEPLOY_MODEL=True to deploy to Model Serving")
if USE_FAST_OPTIMIZATION and OPTIMIZE_AGENT:
    print(f"   ⚡ Run with USE_FAST_OPTIMIZATION=False for production optimization")

print(f"\n🏁 Workflow completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# COMMAND ----------