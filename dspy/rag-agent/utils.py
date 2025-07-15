import os
import json
from typing import Optional

import dspy
from dspy.retrieve.databricks_rm import DatabricksRM


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


def load_optimized_program(program_class, config=None):
    """Load a pre-compiled optimized DSPy program if available.
    
    Args:
        program_class: The DSPy program class to instantiate
        config: ModelConfig object or dictionary containing DSPy configuration
    """
    # Get optimized program path from config or environment
    if config:
        # Handle both ModelConfig and dict types
        try:
            # Try ModelConfig pattern first (no default value)
            dspy_config = config.get("dspy_config") or {}
        except TypeError:
            # Fall back to dictionary pattern (with default value)
            dspy_config = config.get("dspy_config", {})
        optimized_path = os.getenv("DSPY_OPTIMIZED_PROGRAM_PATH") or dspy_config.get("optimized_program_path")
    else:
        optimized_path = os.getenv("DSPY_OPTIMIZED_PROGRAM_PATH")
    
    if optimized_path and os.path.exists(optimized_path):
        try:
            # Try DSPy's native load method first
            if optimized_path.endswith('.json'):
                # Load using DSPy's save/load mechanism
                program = program_class(build_retriever(config))
                program.load(optimized_path)
                return program
            else:
                # Fallback to pickle (less reliable)
                import pickle
                with open(optimized_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Failed to load optimized program: {e}")
            
            # Try loading components
            components_path = optimized_path.replace('.pkl', '_components.json')
            if os.path.exists(components_path):
                try:
                    return load_from_components(components_path, program_class, config)
                except Exception as e2:
                    print(f"Failed to load from components: {e2}")
    
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
    program = program_class(retriever)
    
    # Apply optimized components
    if 'signature' in components and 'instructions' in components['signature']:
        program.response_generator.signature.instructions = components['signature']['instructions']
    
    return program 