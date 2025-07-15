#!/usr/bin/env python3
"""
Example usage of DSPy RAG Agent with different configurations.

This file demonstrates how to use the agent with different config files
for development, production, and custom scenarios.
"""

import mlflow
from agent import DSPyRAGChatAgent
from mlflow.types.agent import ChatAgentMessage

def example_development_usage():
    """Example using default development configuration."""
    print("=== Development Configuration Example ===")
    
    # Uses config.yaml by default
    agent = DSPyRAGChatAgent()
    
    # Test message
    test_message = [
        ChatAgentMessage(
            id="test-1",
            role="user",
            content="What is machine learning?"
        )
    ]
    
    response = agent.predict(test_message)
    print(f"Response: {response.messages[0].content}")
    print()


def example_optimized_usage():
    """Example using optimized production configuration."""
    print("=== Optimized Configuration Example ===")
    
    # Load optimized config
    optimized_config = mlflow.models.ModelConfig(
        development_config="config_optimized.yaml"
    )
    
    # Create agent with optimized config
    agent = DSPyRAGChatAgent(config=optimized_config)
    
    # Test message
    test_message = [
        ChatAgentMessage(
            id="test-2",
            role="user",
            content="Explain neural networks briefly."
        )
    ]
    
    response = agent.predict(test_message)
    print(f"Response: {response.messages[0].content}")
    print()


def example_custom_config():
    """Example using custom configuration dictionary."""
    print("=== Custom Configuration Example ===")
    
    # Custom config as dictionary
    custom_config_dict = {
        "llm_config": {
            "endpoint": "databricks/databricks-claude-3-7-sonnet",
            "max_tokens": 1000,
            "temperature": 0.5,
            "top_p": 0.95
        },
        "vector_search": {
            "top_k": 4
        },
        "agent_config": {
            "use_optimized": False,
            "enable_tracing": True,
            "verbose": True
        }
    }
    
    # Create ModelConfig from dictionary
    custom_config = mlflow.models.ModelConfig(
        development_config=custom_config_dict
    )
    
    # Create agent with custom config
    agent = DSPyRAGChatAgent(config=custom_config)
    
    # Test message
    test_message = [
        ChatAgentMessage(
            id="test-3",
            role="user", 
            content="How does vector search work?"
        )
    ]
    
    response = agent.predict(test_message)
    print(f"Response: {response.messages[0].content}")
    print()


def example_environment_specific():
    """Example showing environment-specific configuration loading."""
    print("=== Environment-Specific Configuration Example ===")
    
    import os
    
    # Determine environment
    env = os.getenv("ENVIRONMENT", "development")
    
    config_files = {
        "development": "config.yaml",
        "production": "config_optimized.yaml",
        "staging": "config.yaml"  # Could have separate staging config
    }
    
    config_file = config_files.get(env, "config.yaml")
    print(f"Using config file: {config_file} for environment: {env}")
    
    # Load environment-specific config
    env_config = mlflow.models.ModelConfig(
        development_config=config_file
    )
    
    agent = DSPyRAGChatAgent(config=env_config)
    print(f"Agent created with {env} configuration")
    print()


if __name__ == "__main__":
    print("DSPy RAG Agent Configuration Examples")
    print("=" * 50)
    
    # Note: These examples assume proper environment variables are set:
    # - VS_INDEX_FULLNAME
    # - DSPY_OPTIMIZED_PROGRAM_PATH (for optimized config)
    
    try:
        example_development_usage()
        example_optimized_usage()
        example_custom_config()
        example_environment_specific()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to set the required environment variables:")
        print("- VS_INDEX_FULLNAME=catalog.schema.index_name")
        print("- DSPY_OPTIMIZED_PROGRAM_PATH=/path/to/optimized/program.pkl (for optimized config)") 