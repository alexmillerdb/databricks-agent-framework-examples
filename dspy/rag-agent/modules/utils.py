import os
import json
from typing import Optional, Tuple, Dict, Any

import dspy
from dspy.retrieve.databricks_rm import DatabricksRM
from dotenv import load_dotenv
from databricks.connect import DatabricksSession

def build_retriever(config=None) -> DatabricksRM:
    """Create a retriever for the Vector Search index using config or env vars.

    Args:
        config: ModelConfig object or dictionary containing vector search configuration
        
    Required config or env vars:
      * VS_INDEX_FULLNAME  – catalog.schema.index_name (Unity Catalog path)
    """
    # Get configuration values, falling back to environment variables
    if config:
        # Handle both ModelConfig and dict types
        try:
            # Try ModelConfig pattern first (no default value)
            vs_config = config.get("vector_search") or {}
        except TypeError:
            # Fall back to dictionary pattern (with default value)
            vs_config = config.get("vector_search", {})
        index_fullname = os.getenv("VS_INDEX_FULLNAME") or vs_config.get("index_fullname")
        text_column_name = vs_config.get("text_column_name", "chunk")
        docs_id_column_name = vs_config.get("docs_id_column_name", "id")
        columns = vs_config.get("columns", ["id", "title", "chunk_id"])
        top_k = int(os.getenv("DSPY_TOP_K", vs_config.get("top_k", 5)))
    else:
        # Fallback to environment variables only
        index_fullname = os.getenv("VS_INDEX_FULLNAME")
        text_column_name = "chunk"
        docs_id_column_name = "id"
        columns = ["id", "title", "chunk_id"]
        top_k = int(os.getenv("DSPY_TOP_K", "5"))

    if not index_fullname:
        raise ValueError(
            "Environment variable VS_INDEX_FULLNAME must be set to the fully-qualified "
            "Vector Search index (e.g. catalog.schema.index_name)"
        )

    return DatabricksRM(
        databricks_index_name=index_fullname,
        text_column_name=text_column_name,
        docs_id_column_name=docs_id_column_name,
        columns=columns,
        k=top_k,
    )


def load_optimized_program(program_class, config=None, mlflow_context=None):
    """Load optimized DSPy program if available.
    
    Args:
        program_class: The DSPy program class to instantiate
        config: ModelConfig object or dictionary containing configuration
        mlflow_context: MLflow context object with artifacts (if available)
        
    Returns:
        Optimized DSPy program or None if not found/loadable
    """
    # Method 1: Try loading from MLflow artifacts first (best practice for deployment)
    if mlflow_context and hasattr(mlflow_context, 'artifacts') and mlflow_context.artifacts:
        if "optimized_program" in mlflow_context.artifacts:
            artifact_path = mlflow_context.artifacts["optimized_program"]
            print(f"🎯 Attempting to load from MLflow artifact: {artifact_path}")
            
            optimized = _load_program_from_path(artifact_path, program_class, config)
            if optimized:
                print("✅ Loaded optimized program from MLflow artifact")
                return optimized
    
    # Method 2: Try loading from config paths
    optimized_path = None
    if config:
        # Handle both ModelConfig and dict types
        try:
            # Look in agent_config first (correct location)
            agent_config = config.get("agent_config") or {}
            optimized_path = agent_config.get("optimized_program_path")
            
            # Fallback to dspy_config for backward compatibility
            if not optimized_path:
                dspy_config = config.get("dspy_config") or {}
                optimized_path = dspy_config.get("optimized_program_path")
        except (TypeError, AttributeError):
            # Fall back to dictionary pattern
            agent_config = config.get("agent_config", {})
            optimized_path = agent_config.get("optimized_program_path")
            
            if not optimized_path:
                dspy_config = config.get("dspy_config", {})
                optimized_path = dspy_config.get("optimized_program_path")
    
    # Method 3: Try environment variable
    if not optimized_path:
        optimized_path = os.getenv("DSPY_OPTIMIZED_PROGRAM_PATH")
    
    # Attempt to load from the path
    if optimized_path:
        print(f"🔍 Attempting to load from config path: {optimized_path}")
        optimized = _load_program_from_path(optimized_path, program_class, config)
        if optimized:
            print("✅ Loaded optimized program from config path")
            return optimized
    
    print("⚠️  No optimized program found via any method")
    return None


def _load_program_from_path(path, program_class, config):
    """
    Helper function to load a DSPy program from a file path.
    
    Args:
        path: File path to the optimized program
        program_class: The DSPy program class to instantiate
        config: Configuration for building retriever
        
    Returns:
        Loaded DSPy program or None
    """
    if not path or not os.path.exists(path):
        print(f"❌ Path does not exist: {path}")
        return None
        
    try:
        # Try DSPy's native load method first
        if path.endswith('.json'):
            # Load using DSPy's save/load mechanism
            program = program_class(build_retriever(config), config)
            program.load(path)
            return program
        else:
            # Fallback to pickle (less reliable)
            import pickle
            with open(path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"❌ Failed to load optimized program from {path}: {e}")
        
        # Try loading components as fallback
        components_path = path.replace('.pkl', '_components.json').replace('.json', '_components.json')
        if os.path.exists(components_path):
            try:
                return load_from_components(components_path, program_class, config)
            except Exception as e2:
                print(f"❌ Failed to load from components: {e2}")
    
    return None


def load_from_components(components_path: str, program_class, config=None):
    """Load program from saved components.
    
    Args:
        components_path: Path to the components JSON file
        program_class: The DSPy program class to instantiate
        config: ModelConfig object or dictionary containing configuration
    """
    with open(components_path, 'r') as f:
        components = json.load(f)
    
    # Create base program
    retriever = build_retriever(config)
    program = program_class(retriever, config)
    
    # Apply optimized components
    if 'signature' in components and 'instructions' in components['signature']:
        program.response_generator.signature.instructions = components['signature']['instructions']
    
    return program 


def setup_environment() -> Tuple[DatabricksSession, str, str]:
    """
    Set up the execution environment for either local development or Databricks.
    
    Returns:
        Tuple of (spark_session, user_name, script_dir)
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


def print_workflow_configuration(
    optimize_agent: bool,
    deploy_model: bool,
    config_file: str,
    eval_dataset_table: str,
    optimization_config: Dict[str, Any]
) -> None:
    """
    Print current workflow configuration for visibility.
    
    Args:
        optimize_agent: Whether to run optimization
        deploy_model: Whether to deploy model
        config_file: Configuration file name
        eval_dataset_table: Evaluation dataset table name
        optimization_config: Optimization configuration dictionary
    """
    print(f"🔧 Workflow Configuration:")
    print(f"  - Optimize Agent: {optimize_agent}")
    print(f"  - Deploy Model: {deploy_model}")
    print(f"  - Config File: {config_file}")
    print(f"  - Eval Dataset: {eval_dataset_table}")
    print(f"  - Optimization Strategy: {optimization_config.get('strategy')}")
    print(f"  - Optimization Level: {optimization_config.get('auto_level')}")
    print(f"  - Training Examples: {optimization_config.get('training_examples_limit')}")
    
    # Print bootstrap config if available
    bootstrap_config = optimization_config.get('bootstrap_config', {})
    if bootstrap_config:
        print(f"  - Bootstrap Demos: {bootstrap_config.get('max_bootstrapped_demos')}")
        print(f"  - Bootstrap Labeled: {bootstrap_config.get('max_labeled_demos')}")
    
    # Print MIPROv2 config if available  
    miprov2_config = optimization_config.get('miprov2_config', {})
    if miprov2_config:
        print(f"  - MIPROv2 Threshold: {miprov2_config.get('metric_threshold')}")
        print(f"  - MIPROv2 Temperature: {miprov2_config.get('init_temperature')}")
    
    print(f"  - Concurrent Threads: {optimization_config.get('num_threads')}")


def load_config_safely(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration file safely with error handling.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary or None if failed to load
    """
    try:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"⚠️  Unsupported config file format: {config_path}")
            return None
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return None
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return None


def validate_config_completeness(config: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Validate that configuration contains all required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, missing_fields)
    """
    required_fields = [
        "llm_config.endpoint",
        "llm_config.max_tokens", 
        "llm_config.temperature",
        "vector_search.index_fullname",
        "vector_search.top_k"
    ]
    
    missing_fields = []
    
    for field in required_fields:
        keys = field.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
        except (KeyError, TypeError):
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields


def get_config_value(config: Dict[str, Any], key_path: str, default=None):
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "llm_config.endpoint")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_directory_if_needed(directory_path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory_path: Path to directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        print(f"📁 Created directory: {directory_path}")
    else:
        print(f"📁 Directory exists: {directory_path}")