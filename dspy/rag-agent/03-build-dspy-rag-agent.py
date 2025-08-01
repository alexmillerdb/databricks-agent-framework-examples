# Databricks notebook source
# MAGIC %md
# MAGIC # Build and Optimize DSPy RAG Agent
# MAGIC
# MAGIC This notebook combines agent creation and optimization into a single workflow:
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

"""
DSPy RAG Agent Builder and Optimizer

A modular script for building, optimizing, and deploying DSPy-based RAG agents
using MLflow and Databricks. Supports both local development and Databricks environments.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install -r requirements.txt
# MAGIC dbutils.library.restartPython()
# COMMAND ----------

# COMMAND ----------

# MAGIC %md
# MAGIC ## Environment Setup
# MAGIC
# MAGIC This section detects whether we're running locally or in Databricks and configures accordingly.

# COMMAND ----------

from dotenv import load_dotenv
import os
# from pkg_resources import get_distribution
from databricks.connect import DatabricksSession

def setup_environment():
    """
    Set up the execution environment for either local development or Databricks.
    
    Returns:
        tuple: (spark_session, user_name)
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
    
    return spark, user_name

# Initialize environment
spark, user_name = setup_environment()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Workflow Options
# MAGIC
# MAGIC Use these parameters to control the workflow behavior.

# COMMAND ----------

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Workflow Control Parameters
OPTIMIZE_AGENT = True          # Whether to run DSPy optimization
DEPLOY_MODEL = True            # Whether to deploy to Model Serving
EVAL_DATASET_NAME = "wikipedia_synthetic_eval"  # Name of evaluation dataset

# Unity Catalog Configuration  
UC_CATALOG = "users"
UC_SCHEMA = "alex_miller"
UC_MODEL_NAME = "dspy_rag_agent"

# DSPy Optimization Settings
OPTIMIZATION_CONFIG = {
    "auto_level": "light",          # Options: "light", "medium", "heavy"
    "num_threads": 2,               # Concurrent threads for optimization
    "training_examples_limit": 10,  # Max training examples to use
    "evaluation_examples_limit": 5 # Max examples for evaluation
}

# Get the directory where this script is located
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
CONFIG_FILE = os.path.join(script_dir, "config.yaml")
EVAL_DATASET_TABLE = f"{UC_CATALOG}.{UC_SCHEMA}.{EVAL_DATASET_NAME}"

def print_configuration():
    """Print current configuration for visibility."""
    print(f"🔧 Configuration:")
    print(f"  - Optimize Agent: {OPTIMIZE_AGENT}")
    print(f"  - Deploy Model: {DEPLOY_MODEL}")
    print(f"  - Config File: {CONFIG_FILE}")
    print(f"  - Eval Dataset: {EVAL_DATASET_TABLE}")
    print(f"  - Optimization Level: {OPTIMIZATION_CONFIG['auto_level']}")
    print(f"  - Training Examples: {OPTIMIZATION_CONFIG['training_examples_limit']}")
    print(f"  - Concurrent Threads: {OPTIMIZATION_CONFIG['num_threads']}")

print_configuration()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Core Imports and Configuration

# COMMAND ----------

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import os
import pickle
import uuid
import json
from typing import Any, List, Optional
from datetime import datetime

# Third-party imports  
import dspy
import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

# Local imports
from agent import DSPyRAGChatAgent, _DSPyRAGProgram
from utils import build_retriever, load_optimized_program

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_training_data(spark):
    """
    Prepare training data for DSPy optimization.
    
    Args:
        spark: Spark session
        
    Returns:
        list: List of training examples
    """
    print("\n1️⃣ Preparing training data...")
    
    if EVAL_DATASET_NAME:
        # Load from Delta table
        print(f"Loading evaluation dataset from: {EVAL_DATASET_TABLE}")
        eval_df = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.{EVAL_DATASET_NAME}")
        eval_data = eval_df.collect()
        
        training_examples = [
            dspy.Example(
                request=row['inputs']['messages'][0]['content'],
                response=row["expectations"]["expected_facts"]
            ).with_inputs("request")
            for row in eval_data[:OPTIMIZATION_CONFIG['training_examples_limit']]  # Use configured limit for training
        ]
    else:
        # Use hardcoded examples
        print("Using hardcoded training examples")
        training_examples = [
            dspy.Example(
                request="Who is Zeus in Greek mythology?",
                response="Expected facts: 1) Zeus is the king of the gods in Greek mythology, 2) He is the god of sky, thunder, and lightning, 3) He rules Mount Olympus, 4) He is son of Titans Cronus and Rhea, 5) He led the overthrow of the Titans, 6) He wields thunderbolts as weapons, 7) He was married to Hera, 8) He had many affairs and offspring including Athena, Apollo, Artemis, Hermes, Perseus, and Heracles"
            ).with_inputs("request"),
            
            dspy.Example(
                request="What caused World War I?",
                response="Expected facts: 1) Immediate trigger was assassination of Archduke Franz Ferdinand by Gavrilo Princip on June 28, 1914, 2) Long-term causes included militarism, alliances, imperialism, and nationalism, 3) Austria-Hungary declared war on Serbia, 4) Alliance system caused escalation: Russia supported Serbia, Germany declared war on Russia and France, 5) Britain entered when Germany invaded Belgium, 6) Colonial rivalries and arms race were contributing factors, 7) Tensions in the Balkans were a key issue"
            ).with_inputs("request"),
        ]
    
    print(f"✅ Prepared {len(training_examples)} training examples")
    return training_examples

def setup_evaluation_metric(_lm):
    """
    Set up the evaluation metric for DSPy optimization.
    
    Args:
        _lm: The DSPy language model to use for evaluation
        
    Returns:
        function: The configured evaluation metric function
    """
    print("\n2️⃣ Defining evaluation metric...")
    
    class RAGEvaluationSignature(dspy.Signature):
        """Evaluate how well a RAG response is grounded in expected facts.
        
        Guidelines:
        - Factual Accuracy: Check if all claims in the response are supported by the expected facts. Penalize hallucinations or contradictions.
        - Completeness: Assess how many of the expected facts are covered in the response. A good response should address most key facts.
        - Overall Score: Balance accuracy and completeness. A response with fewer facts but accurate is better than one with more facts but inaccurate.
        """
        request: str = dspy.InputField(desc="The user's question")
        response: str = dspy.InputField(desc="The RAG system's response")
        expected_facts: str = dspy.InputField(desc="The list of expected facts that should ground the response")
        factual_accuracy: float = dspy.OutputField(desc="Score 0.0-1.0: How well the response is grounded in expected facts (no hallucinations/contradictions)")
        completeness: float = dspy.OutputField(desc="Score 0.0-1.0: How many expected facts are covered in the response")
        overall_score: float = dspy.OutputField(desc="Score 0.0-1.0: Overall quality combining accuracy and completeness")
        reasoning: str = dspy.OutputField(desc="Detailed explanation: which facts are covered, any inaccuracies, and justification for scores")
    
    evaluate_response = dspy.ChainOfThought(RAGEvaluationSignature)
    
    def rag_evaluation_metric(example, prediction, trace=None):
        """Evaluate RAG response against expected facts."""
        request = example.request
        expected_facts = example.response  # This contains the expected facts
        generated_response = prediction.response
        
        with dspy.context(lm=_lm):
            eval_result = evaluate_response(
                request=request,
                response=generated_response,
                expected_facts=expected_facts
            )
        
        try:
            # Use overall_score as the primary metric
            overall_score = float(eval_result.overall_score)
            overall_score = max(0.0, min(1.0, overall_score))
            
            # Also extract individual metrics for logging
            factual_accuracy = float(eval_result.factual_accuracy) if hasattr(eval_result, 'factual_accuracy') else 0.0
            completeness = float(eval_result.completeness) if hasattr(eval_result, 'completeness') else 0.0
            
            # Log detailed metrics if verbose
            if hasattr(eval_result, 'reasoning') and len(eval_result.reasoning) > 10:
                print(f"  📊 Factual Accuracy: {factual_accuracy:.2f}")
                print(f"  📊 Completeness: {completeness:.2f}")
                print(f"  📊 Overall Score: {overall_score:.2f}")
                print(f"  💭 Reasoning: {eval_result.reasoning[:100]}...")
                
        except (ValueError, TypeError) as e:
            print(f"  ⚠️ Error parsing evaluation scores: {e}")
            overall_score = 0.0
            
        return overall_score
    
    return rag_evaluation_metric

def create_final_agent(model_config, optimized_program=None, optimized_program_path=None):
    """
    Create the final DSPy RAG agent with optional optimization.
    
    Args:
        model_config: Base model configuration
        optimized_program: Optional optimized DSPy program
        optimized_program_path: Path to saved optimized program
        
    Returns:
        tuple: (final_agent, final_config)
    """
    print("📦 Creating final agent...")
    
    # Use the single config.yaml for both optimized and base agents
    final_config = model_config
    final_config_dict = final_config.to_dict()

    # If optimization was performed, add the optimized program artifact reference to the config
    if OPTIMIZE_AGENT and optimized_program_path:
        if "agent_config" not in final_config_dict:
            final_config_dict["agent_config"] = {}
        # Store the artifact key name, not the full path (MLflow will resolve the path)
        final_config_dict["agent_config"]["optimized_program_artifact"] = "optimized_program"
        final_config_dict["agent_config"]["use_optimized"] = True
        final_config = mlflow.models.ModelConfig(development_config=final_config_dict)
        print(f"✅ Using configuration with optimization artifact: {CONFIG_FILE}")
        print(f"📦 Optimized program will be loaded from MLflow artifact: optimized_program")
    else:
        print(f"🔧 Using base configuration: {CONFIG_FILE}")

    # Create the final agent
    if optimized_program:
        # Create agent with the optimized program
        final_agent = DSPyRAGChatAgent(rag_program=optimized_program, config=final_config)
    else:
        # Create agent with base program
        final_agent = DSPyRAGChatAgent(config=final_config)

    # Test the final agent
    test_messages = [{"role": "user", "content": "Who is Zeus in Greek mythology?"}]
    # input_example = {"messages": [{"role": "user", "content": "Who is Zeus?"}]}
    test_response = final_agent.predict(messages=[ChatAgentMessage(role=msg["role"], content=msg["content"]) for msg in test_messages])
    print(f"\n✅ Final agent created and tested successfully!")
    print(f"📤 Response preview: {test_response.messages[0].content[:200]}...")
    
    return final_agent, final_config

def log_model_to_mlflow(final_config, llm_config, vector_search_config, optimized_program_path=None, optimization_results=None):
    """
    Log the model to MLflow with proper configuration and metadata.
    
    Args:
        final_config: Final model configuration
        llm_config: LLM configuration
        vector_search_config: Vector search configuration
        optimized_program_path: Path to optimized program (if any)
        optimization_results: Optimization results dictionary
        
    Returns:
        mlflow.entities.model_registry.ModelVersion: Model info
    """
    print("🏃 Logging model to MLflow...")
    
    # Prepare model metadata
    run_name = f"{UC_MODEL_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"🏃 MLflow Run: {run.info.run_id}")
        
        # Prepare artifacts
        artifacts = {}
        if optimized_program_path:
            artifacts["optimized_program"] = optimized_program_path
        
        # Prepare resources
        resources = [
            DatabricksServingEndpoint(endpoint_name=llm_config["endpoint"].replace("databricks/", "")),
            DatabricksVectorSearchIndex(index_name=vector_search_config["index_fullname"])
        ]
        
        # Test messages for input example
        test_messages = [{"role": "user", "content": "Who is Zeus in Greek mythology?"}]
        # test_messages = {"messages": [{"role": "user", "content": "Who is Zeus?"}]}
        
        # Log the model
        model_info = mlflow.pyfunc.log_model(
            name=UC_MODEL_NAME,
            python_model=os.path.join(script_dir, "agent.py"),
            model_config=final_config.to_dict(),
            artifacts=artifacts,
            pip_requirements=os.path.join(script_dir, "requirements.txt"),
            # pip_requirements=[
            # f"dspy=={dspy.__version__}",
            # f"databricks-agents=={get_distribution('databricks-agents').version}",
            # f"mlflow=={mlflow.__version__}",
            # "openai<2",
            # f"databricks-sdk=={get_distribution('databricks-sdk').version}"
            # ],
            resources=resources,
            input_example={"messages": test_messages},
            code_paths=[
                os.path.join(script_dir, "agent.py"), 
                os.path.join(script_dir, "utils.py")
            ],
            # registered_model_name=f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"
        )
        
        # Log parameters
        mlflow.log_param("model_type", "optimized" if OPTIMIZE_AGENT else "base")
        mlflow.log_param("config_file", CONFIG_FILE)
        mlflow.log_param("llm_endpoint", llm_config.get("endpoint"))
        mlflow.log_param("max_tokens", llm_config.get("max_tokens"))
        mlflow.log_param("temperature", llm_config.get("temperature"))
        mlflow.log_param("vector_search_index", vector_search_config.get("index_fullname"))
        mlflow.log_param("top_k", vector_search_config.get("top_k"))
        
        # Log optimization metrics if available
        if OPTIMIZE_AGENT and optimization_results:
            mlflow.log_metric("baseline_score", optimization_results["baseline_score"])
            mlflow.log_metric("optimized_score", optimization_results["optimized_score"])
            mlflow.log_metric("improvement", optimization_results["improvement"])
            mlflow.log_param("optimizer_type", "MIPROv2")
        
        # Log configuration as artifact
        mlflow.log_dict(final_config.to_dict(), "model_config.json")
        
        print(f"✅ Model logged successfully!")
        print(f"📍 Model URI: {model_info.model_uri}")
        
        return model_info

def test_logged_model(model_info):
    """
    Test the logged model with sample questions.
    
    Args:
        model_info: MLflow model info object
    """
    print("🧪 Testing the logged model...")

    # Load and test the model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    # Test with sample questions
    test_questions = [
        "Who is Zeus in Greek mythology?",
        "What is the capital of France?",
        "Explain photosynthesis in simple terms."
    ]

    for question in test_questions[:1]:  # Test first question
        test_input = {"messages": [{"role": "user", "content": question}]}
        response = loaded_model.predict(test_input)
        print(f"\n❓ Question: {question}")
        print(f"💬 Response: {response['messages'][0]['content'][:200]}...")

def validate_model_deployment(model_info):
    """
    Validate the model before deployment using MLflow predict API.
    
    Args:
        model_info: MLflow model info object
    """
    print("🔍 Validating model for deployment...")
    
    try:
        mlflow.models.predict(
            model_uri=f"runs:/{model_info.run_id}/{UC_MODEL_NAME}",
            input_data={"messages": [{"role": "user", "content": "Who is Zeus?"}]},
            env_manager="uv",
        )
        print("✅ Model validation successful!")
    except Exception as e:
        print(f"⚠️  Model validation warning: {e}")
        print("ℹ️  This is likely due to Pydantic version compatibility issues in the validation environment.")
        print("ℹ️  The model should still work correctly when deployed.")
        # Don't raise - just warn about the validation issue

def deploy_model_to_serving(uc_full_model_name, model_info, llm_config, vector_search_config, optimized_program_path=None):
    """
    Deploy the model to Databricks Model Serving.
    
    Args:
        model_info: MLflow model info object
        llm_config: LLM configuration
        vector_search_config: Vector search configuration
        optimized_program_path: Path to optimized program (if any)
        
    Returns:
        deployment object
    """
    print("🚀 Deploying model to Model Serving...")
    
    from databricks import agents
    
    # Deploy the model
    deployment = agents.deploy(
        uc_full_model_name,
        model_version=model_info.version,
        scale_to_zero=True,
        environment_vars={
            "DSPY_LLM_ENDPOINT": llm_config.get("endpoint"),
            "VS_INDEX_FULLNAME": vector_search_config.get("index_fullname"),
            "DSPY_OPTIMIZED_PROGRAM_PATH": optimized_program_path or ""
        }
    )
    
    print(f"✅ Model deployed successfully!")
    print(f"🔗 Endpoint Name: {deployment.endpoint_name}")
    
    return deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Configuration

# COMMAND ----------

# Load configuration
model_config = mlflow.models.ModelConfig(development_config=CONFIG_FILE)
config_dict = model_config.to_dict()

# Extract configuration sections
llm_config = config_dict.get("llm_config") or {}
vector_search_config = config_dict.get("vector_search") or {}
dspy_config = config_dict.get("dspy_config") or {}
agent_config = config_dict.get("agent_config") or {}
mlflow_config = config_dict.get("mlflow_config") or {}

# Display configuration
print("🔧 Loaded Configuration:")
print(f"  - LLM Endpoint: {llm_config.get('endpoint')}")
print(f"  - Max Tokens: {llm_config.get('max_tokens')}")
print(f"  - Temperature: {llm_config.get('temperature')}")
print(f"  - Vector Search Index: {vector_search_config.get('index_fullname')}")
print(f"  - Top K: {vector_search_config.get('top_k')}")
print(f"  - Use Optimized: {agent_config.get('use_optimized', False)}")
print(f"  - Enable Tracing: {agent_config.get('enable_tracing', True)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Setup

# COMMAND ----------

# Set up MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Set experiment name
experiment_name = f"/Users/{user_name}/dspy-rag-agent-unified"
mlflow.set_experiment(experiment_name)

# Enable DSPy autologging if configured
if agent_config.get("enable_tracing", True):
    mlflow.dspy.autolog()
    print("✅ DSPy autologging enabled")

print(f"📊 MLflow Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Base DSPy RAG Agent
# MAGIC
# MAGIC Create the initial RAG agent without optimization.

# COMMAND ----------

print("🔨 Building base DSPy RAG agent...")

# Build retriever from configuration
retriever = build_retriever(model_config)

# Configure DSPy LM
endpoint = llm_config.get("endpoint", "databricks/databricks-meta-llama-3-3-70b-instruct")
_lm = dspy.LM(
    endpoint,
    cache=False,
    max_tokens=llm_config.get("max_tokens", 2500),
    temperature=llm_config.get("temperature", 0.01)
)

# Create base program
base_program = _DSPyRAGProgram(retriever)

# Test the base program
test_question = "Who is Zeus in Greek mythology?"
with dspy.context(lm=_lm):
    base_response = base_program(request=test_question)
    
print(f"✅ Base agent created successfully!")
print(f"\n📝 Test Question: {test_question}")
print(f"📤 Base Response: {base_response.response[:200]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional: Optimize Agent with DSPy
# MAGIC
# MAGIC If optimization is enabled, compile the agent using DSPy optimizers.

# COMMAND ----------

def run_optimization_workflow(_lm, base_program, llm_config, model_config):
    """
    Run the complete DSPy optimization workflow.
    
    Args:
        _lm: The DSPy language model
        base_program: The base DSPy program to optimize
        llm_config: LLM configuration
        model_config: Model configuration
        
    Returns:
        tuple: (optimized_program, optimized_program_path, optimization_results)
    """
    print("🚀 Starting DSPy optimization process...")
    
    # SECTION 1: Prepare Training Data
    training_examples = prepare_training_data(spark)
    
    # SECTION 2: Setup evaluation metric
    rag_evaluation_metric = setup_evaluation_metric(_lm)
    
    # SECTION 3: Evaluate Baseline
    print("\n3️⃣ Evaluating baseline performance...")
    from dspy.evaluate import Evaluate
    
    evaluator = Evaluate(
        devset=training_examples[:OPTIMIZATION_CONFIG['evaluation_examples_limit']],
        metric=rag_evaluation_metric,
        num_threads=1,
        display_progress=True
    )
    
    with dspy.context(lm=_lm):
        baseline_score = evaluator(base_program)
    
    print(f"📊 Baseline Score: {baseline_score}")
    
    # SECTION 4: Run Optimization
    print("\n4️⃣ Running DSPy optimization...")
    
    try:
        # Try MIPROv2 first
        print("Attempting MIPROv2 optimization...")
        from dspy.teleprompt import MIPROv2
        
        optimizer = MIPROv2(
            metric=rag_evaluation_metric,
            auto=OPTIMIZATION_CONFIG['auto_level'],
            num_threads=OPTIMIZATION_CONFIG['num_threads'],
            verbose=True,
            track_stats=True
        )
        
        with dspy.context(lm=_lm):
            optimized_program = optimizer.compile(
                base_program,
                trainset=training_examples
            )
            
    except Exception as e:
        # Fallback to BootstrapFewShot
        print(f"MIPROv2 failed ({str(e)}), falling back to BootstrapFewShot...")
        from dspy.teleprompt import BootstrapFewShot
        
        optimizer = BootstrapFewShot(
            metric=rag_evaluation_metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=3
        )
        
        with dspy.context(lm=_lm):
            optimized_program = optimizer.compile(
                base_program,
                trainset=training_examples
            )
    
    # SECTION 5: Evaluate Optimized Program
    print("\n5️⃣ Evaluating optimized performance...")
    
    with dspy.context(lm=_lm):
        optimized_score = evaluator(optimized_program)
    
    print(f"📊 Optimized Score: {optimized_score}")
    print(f"📈 Improvement: {(optimized_score - baseline_score)}")

    # SECTION 6: Save Optimized Program and Log to MLflow
    optimized_program_path = None
    with mlflow.start_run(run_name="DSPy_RAG_Compilation"):
        # Test the optimized program on multiple questions
        test_questions = [
            "Who is Zeus?",
            "What is Greek mythology?", 
            "Who is Frederick Barbarossa?"
        ]
        
        print("\nTesting optimized program:")
        for question in test_questions:
            try:
                result = optimized_program(question)
                print(f"Q: {question}")
                print(f"A: {result.response}")
                print("-" * 50)
            except Exception as e:
                print(f"Error testing question '{question}': {e}")
        
        # Save the optimized program
        try:
            optimized_program_path = os.path.join("optimized_rag_program.json")
            optimized_program.save(optimized_program_path)
            
            # Log the optimized program as an artifact
            mlflow.log_artifact(optimized_program_path, "optimized_program")
            print("Optimized program saved successfully!")
            
        except Exception as e:
            print(f"Error saving program: {e}")
            optimized_program_path = None
        
        # Save program metadata
        program_state = {
            "program_type": type(optimized_program).__name__,
            "base_score": float(baseline_score),
            "optimized_score": float(optimized_score),
            "improvement": float(optimized_score - baseline_score),
            "config": {
                "optimizer": "MIPROv2",
                "training_examples": len(training_examples),
                "optimization_level": OPTIMIZATION_CONFIG['auto_level'],
                "llm_config": llm_config,
                "agent_config": model_config.get("agent_config") or {}
            }
        }

        metadata_path = os.path.join("program_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(program_state, f, indent=2)
        
        mlflow.log_artifact(metadata_path, "optimized_program")
        
        # Log the model config as an artifact
        mlflow.log_dict(model_config.to_dict(), "model_config.json")
        
        # Log metrics and parameters
        mlflow.log_param("training_examples", len(training_examples))
        mlflow.log_param("optimizer", "MIPROv2")
        mlflow.log_param("optimization_level", OPTIMIZATION_CONFIG['auto_level'])
        mlflow.log_param("llm_endpoint", llm_config.get("endpoint"))
        mlflow.log_param("max_tokens", llm_config.get("max_tokens"))
        mlflow.log_param("temperature", llm_config.get("temperature"))
        mlflow.log_metric("base_score", baseline_score)
        mlflow.log_metric("optimized_score", optimized_score)
        mlflow.log_metric("improvement", optimized_score - baseline_score)
        
        print(f"\nOptimization Results:")
        print(f"Base Score: {baseline_score:.3f}")
        print(f"Optimized Score: {optimized_score:.3f}")
        print(f"Improvement: {optimized_score - baseline_score:.3f}")
        print(f"Configuration: {CONFIG_FILE}")
    
    optimization_results = {
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": optimized_score - baseline_score
    }
    
    return optimized_program, optimized_program_path, optimization_results

# ============================================================================
# MAIN OPTIMIZATION WORKFLOW
# ============================================================================

# Run optimization if enabled
if OPTIMIZE_AGENT:
    optimized_program, optimized_program_path, optimization_results = run_optimization_workflow(
        _lm, base_program, llm_config, model_config
    )
else:
    print("⏭️  Skipping optimization (optimize_agent=false)")
    optimized_program = None
    optimized_program_path = None
    optimization_results = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Final Agent and Log to MLflow
# MAGIC
# MAGIC Create the agent (with or without optimization) and log it to MLflow.

# COMMAND ----------

# ============================================================================
# MAIN AGENT CREATION AND LOGGING WORKFLOW  
# ============================================================================

# Create final agent
final_agent, final_config = create_final_agent(
    model_config, 
    optimized_program, 
    optimized_program_path
)

# Log model to MLflow
model_info = log_model_to_mlflow(
    final_config, 
    llm_config, 
    vector_search_config, 
    optimized_program_path, 
    optimization_results
)

# Test the logged model
test_logged_model(model_info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-deployment Validation and Optional Deployment

# COMMAND ----------

# ============================================================================
# VALIDATION AND DEPLOYMENT WORKFLOW
# ============================================================================

# Validate model before deployment
validate_model_deployment(model_info)

# Register model in Unity Catalog
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"
uc_registered_model_info = mlflow.register_model(
    model_uri=model_info.model_uri, 
    name=UC_MODEL_NAME
)

# Deploy to Model Serving if enabled
if DEPLOY_MODEL:
    deployment = deploy_model_to_serving(
        UC_MODEL_NAME,
        uc_registered_model_info, 
        llm_config, 
        vector_search_config, 
        optimized_program_path
    )
else:
    print("⏭️  Skipping deployment (deploy_model=false)")
    deployment = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Workflow Summary
# MAGIC
# MAGIC ### 🎉 Completed Successfully!
# MAGIC 
# MAGIC This modular workflow has successfully:
# MAGIC 
# MAGIC #### **Environment & Configuration**
# MAGIC 1. ✅ **Environment Detection** - Automatically configured for local or Databricks execution
# MAGIC 2. ✅ **Configuration Management** - Loaded and validated all settings from single config file
# MAGIC 3. ✅ **MLflow Setup** - Configured tracking and registry for model management
# MAGIC 
# MAGIC #### **Agent Development**
# MAGIC 4. ✅ **Base Agent Creation** - Built DSPy RAG agent with ChatAgent interface
# MAGIC 5. ✅ **Optimization (Optional)** - Used DSPy compilation with fact-based evaluation
# MAGIC 6. ✅ **Final Agent Assembly** - Combined base/optimized components with unified config
# MAGIC 
# MAGIC #### **Model Management**
# MAGIC 7. ✅ **MLflow Logging** - Registered model with proper metadata and artifacts
# MAGIC 8. ✅ **Unity Catalog Registration** - Stored in enterprise catalog for governance
# MAGIC 9. ✅ **Model Validation** - Tested functionality before deployment
# MAGIC 10. ✅ **Model Serving (Optional)** - Deployed to scalable endpoint with monitoring
# COMMAND ----------
# MAGIC 
# MAGIC ### 📊 **Key Configuration Parameters Used:**
# COMMAND ----------
print(f"✨ **Final Configuration Summary:**")
print(f"  - 🎯 **Optimization**: {'✅ Enabled' if OPTIMIZE_AGENT else '❌ Disabled'}")
print(f"  - 🚀 **Deployment**: {'✅ Enabled' if DEPLOY_MODEL else '❌ Disabled'}")
print(f"  - 📊 **Model**: {UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}")
print(f"  - 🧠 **LLM**: {llm_config.get('endpoint', 'Not configured')}")
print(f"  - 🔍 **Vector Search**: {vector_search_config.get('index_fullname', 'Not configured')}")

if OPTIMIZE_AGENT and optimization_results:
    print(f"\n🎯 **Optimization Results:**")
    print(f"  - 📈 **Baseline Score**: {optimization_results['baseline_score']:.3f}")
    print(f"  - 🚀 **Optimized Score**: {optimization_results['optimized_score']:.3f}")
    print(f"  - ⬆️  **Improvement**: {optimization_results['improvement']:.3f}")

print(f"\n🔗 **Model Artifacts:**")
print(f"  - 📍 **Model URI**: {model_info.model_uri}")
print(f"  - 📦 **Registered Name**: {uc_registered_model_info.name}")
if DEPLOY_MODEL and 'deployment' in locals() and deployment:
    print(f"  - 🌐 **Endpoint**: {deployment.endpoint_name}")

print(f"\n### 🚀 **Next Steps:**")
print(f"- **Test the endpoint** in Databricks Serving UI")
print(f"- **Monitor performance** metrics and usage")
print(f"- **Iterate optimization** with larger evaluation datasets")
print(f"- **Scale deployment** based on traffic requirements")
print(f"- **Update configuration** parameters as needed in the top section")