#!/usr/bin/env python3
"""
Simple test script for DSPy RAG agent architectures with Databricks integration.
Allows rapid testing and iteration of new agent designs locally while maintaining
connection to Databricks services (Vector Search, Model Serving endpoints).
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Optional

# Suppress logging warnings and errors unless tests actually fail
logging.getLogger("pyspark").setLevel(logging.CRITICAL)
logging.getLogger("py4j").setLevel(logging.CRITICAL)
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("dspy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Add the dspy/rag-agent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "dspy" / "rag-agent"))

def setup_environment():
    """Set up environment for local development with Databricks connection."""
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Detect environment
    try:
        from databricks.connect import DatabricksSession
        
        if os.getenv('DATABRICKS_TOKEN') and os.getenv('DATABRICKS_HOST'):
            print("🏠 Local Development Mode Detected")
            
            # Initialize Databricks Connect for local development
            spark = DatabricksSession.builder.getOrCreate()
            user_name = "alex.miller@databricks.com"
            
            # Set required environment variables if not already set
            if not os.getenv('VS_INDEX_FULLNAME'):
                os.environ['VS_INDEX_FULLNAME'] = "users.alex_miller.wikipedia_chunks_index"
                
        else:
            print("☁️  Databricks Environment Mode")
            import pyspark.sql
            spark = pyspark.sql.SparkSession.getActiveSession()
            
            # Would get user from dbutils in actual Databricks environment
            user_name = "databricks_user"
            
    except ImportError:
        print("ℹ️  Running without Databricks connection")
        user_name = "local_user"
        
        # Set default environment variables for testing
        if not os.getenv('VS_INDEX_FULLNAME'):
            os.environ['VS_INDEX_FULLNAME'] = "users.alex_miller.wikipedia_chunks_index"
    
    print(f"👤 User: {user_name}")
    return user_name

def test_basic_agent():
    """Test the basic DSPyRAGChatAgent functionality."""
    print("\n🧪 Testing Basic DSPyRAGChatAgent...")
    
    try:
        from agent import DSPyRAGChatAgent
        from mlflow.types.agent import ChatAgentMessage
        
        # Initialize agent with default config
        agent = DSPyRAGChatAgent()
        
        # Test with a simple question
        test_messages = [
            ChatAgentMessage(role="user", content="When and where did the heavy metal genre develop?")
        ]
        
        print("📤 Sending test question: 'When and where did the heavy metal genre develop?'")
        response = agent.predict(test_messages)
        
        print("📨 Response received:")
        print(f"   Role: {response.messages[0].role}")
        print(f"   Content: {response.messages[0].content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic agent test failed: {e}")
        return False

def test_custom_program():
    """Test the _DSPyRAGProgram directly for rapid iteration."""
    print("\n🔬 Testing _DSPyRAGProgram directly...")
    
    try:
        from agent import _DSPyRAGProgram
        from utils import build_retriever
        import mlflow
        
        # Load configuration
        config_path = Path(__file__).parent / "dspy" / "rag-agent" / "config.yaml"
        config = mlflow.models.ModelConfig(development_config=str(config_path))
        
        # Build retriever
        retriever = build_retriever(config)
        
        # Create program
        program = _DSPyRAGProgram(retriever)
        
        # Test with a direct query
        test_query = "What is the capital of France?"
        print(f"📤 Testing query: '{test_query}'")
        
        result = program(test_query)
        
        print("📨 Direct Program Response:")
        print(f"   Response: {result.response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom program test failed: {e}")
        return False

def test_streaming():
    """Test streaming functionality."""
    print("\n🌊 Testing Streaming Response...")
    
    try:
        from agent import DSPyRAGChatAgent
        from mlflow.types.agent import ChatAgentMessage
        
        # Initialize agent
        agent = DSPyRAGChatAgent()
        
        # Test streaming
        test_messages = [
            ChatAgentMessage(role="user", content="Explain quantum computing briefly.")
        ]
        
        print("📤 Sending streaming test question...")
        
        for chunk in agent.predict_stream(test_messages):
            print(f"📡 Received streaming chunk:")
            print(f"   Content: {chunk.delta.content}")
            break  # Just test first chunk for demo
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False

def test_config_field_mapping():
    """Test that the program correctly uses config field mappings."""
    print("\n🔧 Testing Config Field Mapping...")
    
    try:
        from agent import _DSPyRAGProgram
        from utils import build_retriever
        import mlflow
        
        # Load configuration
        config_path = Path(__file__).parent / "dspy" / "rag-agent" / "config.yaml"
        config = mlflow.models.ModelConfig(development_config=str(config_path))
        
        # Build retriever and program
        retriever = build_retriever(config)
        program = _DSPyRAGProgram(retriever, config)
        
        # Verify config field mapping
        vs_config = config.get("vector_search") or {}
        expected_text_field = vs_config.get("text_column_name", "chunk")
        expected_id_field = vs_config.get("docs_id_column_name", "id")
        expected_columns = vs_config.get("columns", ["id", "title", "chunk_id"])
        
        print(f"📋 Expected text field: {expected_text_field}")
        print(f"📋 Expected ID field: {expected_id_field}")
        print(f"📋 Expected columns: {expected_columns}")
        
        print(f"✅ Program text field: {program.text_field}")
        print(f"✅ Program ID field: {program.id_field}")
        print(f"✅ Program columns: {program.columns}")
        print(f"✅ Program metadata fields: {program.metadata_fields}")
        
        # Verify fields match config
        assert program.text_field == expected_text_field, f"Text field mismatch: {program.text_field} != {expected_text_field}"
        assert program.id_field == expected_id_field, f"ID field mismatch: {program.id_field} != {expected_id_field}"
        assert program.columns == expected_columns, f"Columns mismatch: {program.columns} != {expected_columns}"
        
        return True
        
    except Exception as e:
        print(f"❌ Config field mapping test failed: {e}")
        return False

def test_custom_config():
    """Test agent with custom configuration."""
    print("\n⚙️ Testing Custom Configuration...")
    
    try:
        from agent import DSPyRAGChatAgent
        from mlflow.types.agent import ChatAgentMessage
        import mlflow
        import tempfile
        import yaml
        
        # Create custom config with different field names
        custom_config = {
            "llm_config": {
                "endpoint": "databricks/databricks-meta-llama-3-1-8b-instruct",
                "max_tokens": 1500,
                "temperature": 0.0
            },
            "vector_search": {
                "index_fullname": "users.alex_miller.wikipedia_chunks_index",
                "text_column_name": "content",  # Different from default
                "docs_id_column_name": "doc_id",  # Different from default
                "columns": ["doc_id", "title", "content"],  # Different structure
                "top_k": 3
            },
            "agent_config": {
                "use_optimized": False,
                "enable_tracing": False
            }
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(custom_config, f)
            custom_config_path = f.name
        
        try:
            # Create agent with custom config
            custom_model_config = mlflow.models.ModelConfig(development_config=custom_config_path)
            agent = DSPyRAGChatAgent(config=custom_model_config)
            
            # Verify the program uses custom field names
            print(f"✅ Custom text field: {agent.rag.text_field}")
            print(f"✅ Custom ID field: {agent.rag.id_field}")
            print(f"✅ Custom columns: {agent.rag.columns}")
            
            # Test with a question
            test_messages = [
                ChatAgentMessage(role="user", content="What is heavy metal music?")
            ]
            
            print("📤 Testing with custom config...")
            response = agent.predict(test_messages)
            print(f"📨 Custom config response received (truncated): {response.messages[0].content[:100]}...")
            
            return True
            
        finally:
            # Cleanup
            import os
            os.unlink(custom_config_path)
        
    except Exception as e:
        print(f"❌ Custom config test failed: {e}")
        return False

def test_field_fallbacks():
    """Test fallback mechanisms for missing fields."""
    print("\n🔄 Testing Field Fallbacks...")
    
    try:
        from agent import _DSPyRAGProgram
        from utils import build_retriever
        import mlflow
        
        # Create config with minimal field specifications
        minimal_config = mlflow.models.ModelConfig()
        minimal_config._config = {
            "vector_search": {
                "index_fullname": "users.alex_miller.wikipedia_chunks_index"
                # Missing text_column_name and other fields - should use defaults
            }
        }
        
        # Build program with minimal config
        retriever = build_retriever(minimal_config)
        program = _DSPyRAGProgram(retriever, minimal_config)
        
        # Verify defaults are used
        print(f"✅ Default text field: {program.text_field}")
        print(f"✅ Default ID field: {program.id_field}")
        print(f"✅ Default columns: {program.columns}")
        
        # Should use fallback defaults
        assert program.text_field == "chunk", f"Expected default 'chunk', got {program.text_field}"
        assert program.id_field == "id", f"Expected default 'id', got {program.id_field}"
        
        return True
        
    except Exception as e:
        print(f"❌ Field fallbacks test failed: {e}")
        return False

def test_metadata_extraction():
    """Test metadata extraction logic."""
    print("\n📊 Testing Metadata Extraction...")
    
    try:
        from agent import _DSPyRAGProgram
        from utils import build_retriever
        import mlflow
        
        # Load configuration
        config_path = Path(__file__).parent / "dspy" / "rag-agent" / "config.yaml"
        config = mlflow.models.ModelConfig(development_config=str(config_path))
        
        # Build program
        retriever = build_retriever(config)
        program = _DSPyRAGProgram(retriever, config)
        
        # Test metadata field extraction
        vs_config = config.get("vector_search") or {}
        text_field = vs_config.get("text_column_name", "chunk")
        id_field = vs_config.get("docs_id_column_name", "id")
        all_columns = vs_config.get("columns", ["id", "title", "chunk_id"])
        
        expected_metadata = [col for col in all_columns if col not in [text_field, id_field]]
        
        print(f"✅ All columns: {all_columns}")
        print(f"✅ Text field: {text_field}")
        print(f"✅ ID field: {id_field}")
        print(f"✅ Expected metadata fields: {expected_metadata}")
        print(f"✅ Actual metadata fields: {program.metadata_fields}")
        
        assert program.metadata_fields == expected_metadata, f"Metadata extraction failed: {program.metadata_fields} != {expected_metadata}"
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata extraction test failed: {e}")
        return False

def test_query_rewriter():
    """Test query rewriting functionality."""
    print("\n🔄 Testing Query Rewriter...")
    
    try:
        from agent import _DSPyRAGProgram
        from utils import build_retriever
        import mlflow
        
        # Load configuration with query rewriter enabled
        config_path = Path(__file__).parent / "dspy" / "rag-agent" / "config.yaml"
        config = mlflow.models.ModelConfig(development_config=str(config_path))
        
        # Build program
        retriever = build_retriever(config)
        program = _DSPyRAGProgram(retriever, config)
        
        # Verify query rewriter is enabled
        agent_config = config.get("agent_config") or {}
        expected_rewriter_enabled = agent_config.get("use_query_rewriter", True)
        
        print(f"✅ Config query rewriter setting: {expected_rewriter_enabled}")
        print(f"✅ Program query rewriter enabled: {program.use_query_rewriter}")
        print(f"✅ Query rewriter module present: {hasattr(program, 'query_rewriter')}")
        
        assert program.use_query_rewriter == expected_rewriter_enabled, f"Query rewriter config mismatch"
        
        if program.use_query_rewriter:
            assert hasattr(program, 'query_rewriter'), "Query rewriter module missing when enabled"
            
            # Test with a simple question to see query rewriting in action
            print("📤 Testing query rewriting with: 'Who started heavy metal music?'")
            result = program("Who started heavy metal music?")
            print(f"📨 Got response: {result.response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Query rewriter test failed: {e}")
        return False

def test_query_rewriter_disabled():
    """Test program behavior with query rewriter disabled."""
    print("\n🚫 Testing Query Rewriter Disabled...")
    
    try:
        from agent import _DSPyRAGProgram
        from utils import build_retriever
        import mlflow
        import tempfile
        import yaml
        
        # Create config with query rewriter disabled
        config_data = {
            "vector_search": {
                "index_fullname": "users.alex_miller.wikipedia_chunks_index",
                "text_column_name": "chunk",
                "docs_id_column_name": "id",
                "columns": ["id", "title", "chunk_id"],
                "top_k": 5
            },
            "agent_config": {
                "use_query_rewriter": False,  # Disabled
                "use_optimized": False,
                "enable_tracing": False
            }
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Create program with query rewriter disabled
            config = mlflow.models.ModelConfig(development_config=config_path)
            retriever = build_retriever(config)
            program = _DSPyRAGProgram(retriever, config)
            
            print(f"✅ Query rewriter disabled: {not program.use_query_rewriter}")
            print(f"✅ Query rewriter module absent: {not hasattr(program, 'query_rewriter')}")
            
            assert not program.use_query_rewriter, "Query rewriter should be disabled"
            assert not hasattr(program, 'query_rewriter'), "Query rewriter module should not exist when disabled"
            
            return True
            
        finally:
            # Cleanup
            import os
            os.unlink(config_path)
        
    except Exception as e:
        print(f"❌ Query rewriter disabled test failed: {e}")
        return False

def interactive_test():
    """Interactive testing mode for rapid iteration."""
    print("\n💬 Interactive Testing Mode")
    print("Type 'quit' to exit, 'help' for commands")
    
    try:
        from agent import DSPyRAGChatAgent, _DSPyRAGProgram
        from mlflow.types.agent import ChatAgentMessage
        from utils import build_retriever
        import mlflow
        
        # Setup
        config_path = Path(__file__).parent / "dspy" / "rag-agent" / "config.yaml"
        config = mlflow.models.ModelConfig(development_config=str(config_path))
        
        # Choose mode
        print("\nSelect mode:")
        print("1. ChatAgent (full MLflow interface)")
        print("2. Direct Program (for rapid iteration)")
        
        mode = input("Enter choice (1 or 2): ").strip()
        
        if mode == "1":
            agent = DSPyRAGChatAgent()
            test_func = lambda q: agent.predict([ChatAgentMessage(role="user", content=q)])
        else:
            retriever = build_retriever(config)
            program = _DSPyRAGProgram(retriever)
            test_func = lambda q: program(q)
        
        while True:
            try:
                question = input("\n🤔 Ask a question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    print("Commands: 'quit' to exit, or ask any question")
                    continue
                elif not question:
                    continue
                
                print("🤖 Thinking...")
                
                if mode == "1":
                    response = test_func(question)
                    print(f"💬 Response: {response.messages[0].content}")
                else:
                    result = test_func(question)
                    print(f"💬 Response: {result.response}")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Interactive mode setup failed: {e}")

def main():
    """Main test runner."""
    print("🚀 DSPy RAG Agent Test Suite")
    print("=" * 40)
    
    # Setup environment
    user_name = setup_environment()
    
    # Run tests
    tests = [
        test_basic_agent,
        test_custom_program,
        test_streaming,
        test_config_field_mapping,
        test_custom_config,
        test_field_fallbacks,
        test_metadata_extraction,
        test_query_rewriter,
        test_query_rewriter_disabled,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} passed")
    
    # Offer interactive mode
    try:
        if input("\n🤔 Would you like to try interactive mode? (y/n): ").lower().startswith('y'):
            interactive_test()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Exiting...")
    
    print("\n✅ Test suite completed!")

if __name__ == "__main__":
    main()