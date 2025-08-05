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

# Add the parent directory (dspy/rag-agent) to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        from modules.utils import build_retriever
        import mlflow
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        config = mlflow.models.ModelConfig(development_config=str(config_path))
        
        # Build retriever
        retriever = build_retriever(config)
        
        # Create program
        program = _DSPyRAGProgram(retriever, config)
        
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
        from modules.utils import build_retriever
        import mlflow
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
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
        from modules.utils import build_retriever
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
        from modules.utils import build_retriever
        import mlflow
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
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
        from modules.utils import build_retriever
        import mlflow
        
        # Load configuration with query rewriter enabled
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
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
        from modules.utils import build_retriever
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

def test_metrics_module():
    """Test the comprehensive metrics module."""
    print("\n📊 Testing Metrics Module...")
    
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "dspy" / "rag-agent"))
        
        from modules.metrics import (
            CitationAccuracyMetric, SemanticF1Metric, CompletenessMetric,
            EndToEndRAGMetric, get_comprehensive_metric
        )
        import dspy
        
        # Create test examples
        example = dspy.Example(
            request="What is heavy metal music?",
            response="Heavy metal is a genre of rock music characterized by aggressive sounds."
        ).with_inputs("request")
        
        # Test prediction with citations
        pred_with_citations = dspy.Prediction(
            response="Heavy metal is a genre of rock music [1]. It developed in the late 1960s [2]."
        )
        
        # Test prediction without citations
        pred_no_citations = dspy.Prediction(
            response="Heavy metal is a genre of rock music that developed in the late 1960s."
        )
        
        # Test Citation Accuracy Metric
        citation_metric = CitationAccuracyMetric()
        citation_score_with = citation_metric(example, pred_with_citations)
        citation_score_without = citation_metric(example, pred_no_citations)
        
        print(f"✅ Citation with citations: {citation_score_with}")
        print(f"✅ Citation without citations: {citation_score_without}")
        
        # Test Semantic F1 Metric
        semantic_metric = SemanticF1Metric()
        semantic_score = semantic_metric(example, pred_with_citations)
        
        print(f"✅ Semantic F1 score: {semantic_score}")
        
        # Test Completeness Metric
        completeness_metric = CompletenessMetric()
        completeness_score = completeness_metric(example, pred_with_citations)
        
        print(f"✅ Completeness score: {completeness_score}")
        
        # Test End-to-End Metric
        end_to_end_metric = get_comprehensive_metric()
        overall_score = end_to_end_metric(example, pred_with_citations)
        
        print(f"✅ End-to-end score: {overall_score}")
        
        # Verify all metrics return reasonable values
        assert isinstance(citation_score_with, bool), "Citation metric should return boolean"
        assert isinstance(semantic_score, float), "Semantic metric should return float"
        assert isinstance(completeness_score, (bool, float)), "Completeness metric should return bool or float"
        assert isinstance(overall_score, float), "End-to-end metric should return float"
        assert 0 <= overall_score <= 1, "End-to-end score should be between 0 and 1"
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics module test failed: {e}")
        return False

def test_training_examples_generation():
    """Test generation of training examples for optimization."""
    print("\n🎯 Testing Training Examples Generation...")
    
    try:
        import dspy
        
        # Test creating training examples in DSPy format
        training_examples = [
            dspy.Example(
                request="When did heavy metal music start?",
                response="Heavy metal music started in the late 1960s and early 1970s in the United Kingdom and United States [1]."
            ).with_inputs("request"),
            
            dspy.Example(
                request="Who are famous heavy metal bands?",
                response="Famous heavy metal bands include Black Sabbath, Led Zeppelin, Deep Purple, and Iron Maiden [1][2]."
            ).with_inputs("request"),
            
            dspy.Example(
                request="What instruments are used in heavy metal?",
                response="Heavy metal typically uses electric guitars, bass guitar, drums, and vocals, often with guitar solos and powerful amplification [1]."
            ).with_inputs("request")
        ]
        
        print(f"✅ Created {len(training_examples)} training examples")
        
        # Verify example structure
        for i, example in enumerate(training_examples):
            assert hasattr(example, 'request'), f"Example {i} missing request"
            assert hasattr(example, 'response'), f"Example {i} missing response"
            assert example.request, f"Example {i} has empty request"
            assert example.response, f"Example {i} has empty response"
            print(f"   Example {i+1}: ✅ Valid structure")
        
        # Test example with query rewriter
        query_rewriter_example = dspy.Example(
            original_question="Who started heavy metal?",
            rewritten_query="founders and originators of heavy metal music genre pioneers musicians"
        ).with_inputs("original_question")
        
        assert hasattr(query_rewriter_example, 'original_question'), "Query rewriter example missing original_question"
        assert hasattr(query_rewriter_example, 'rewritten_query'), "Query rewriter example missing rewritten_query"
        
        print("✅ Query rewriter example structure valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Training examples generation test failed: {e}")
        return False

def test_optimization_config():
    """Test optimization configuration management."""
    print("\n⚙️ Testing Optimization Configuration...")
    
    try:
        import mlflow
        import tempfile
        import yaml
        
        # Create optimization config
        optimization_config = {
            "optimization": {
                "use_miprov2": True,
                "use_bootstrap": True,
                "miprov2_config": {
                    "num_trials": 25,  # Reduced for testing
                    "init_temperature": 1.0,
                    "verbose": True
                },
                "bootstrap_config": {
                    "max_bootstrapped_demos": 4,
                    "max_labeled_demos": 2,
                    "num_candidate_programs": 5,
                    "metric_threshold": 0.6
                },
                "training_examples_limit": 10,
                "validation_examples_limit": 5
            },
            "vector_search": {
                "index_fullname": "users.alex_miller.wikipedia_chunks_index",
                "text_column_name": "chunk",
                "docs_id_column_name": "id",
                "columns": ["id", "title", "chunk_id"],
                "top_k": 3  # Reduced for testing
            },
            "agent_config": {
                "use_query_rewriter": True,
                "use_optimized": False,  # Start with base program
                "enable_tracing": False  # Disable for optimization
            }
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(optimization_config, f)
            config_path = f.name
        
        try:
            # Load and validate config
            config = mlflow.models.ModelConfig(development_config=config_path)
            
            # Test config access
            opt_config = config.get("optimization") or {}
            print(f"✅ Use MIPROv2: {opt_config.get('use_miprov2', False)}")
            print(f"✅ Use Bootstrap: {opt_config.get('use_bootstrap', False)}")
            print(f"✅ Training limit: {opt_config.get('training_examples_limit', 0)}")
            
            mipro_config = opt_config.get("miprov2_config") or {}
            print(f"✅ MIPROv2 trials: {mipro_config.get('num_trials', 0)}")
            
            bootstrap_config = opt_config.get("bootstrap_config") or {}
            print(f"✅ Bootstrap demos: {bootstrap_config.get('max_bootstrapped_demos', 0)}")
            
            # Verify required fields
            assert opt_config.get("training_examples_limit", 0) > 0, "Training examples limit should be positive"
            assert mipro_config.get("num_trials", 0) > 0, "MIPROv2 trials should be positive"
            assert bootstrap_config.get("max_bootstrapped_demos", 0) > 0, "Bootstrap demos should be positive"
            
            return True
            
        finally:
            # Cleanup
            import os
            os.unlink(config_path)
        
    except Exception as e:
        print(f"❌ Optimization config test failed: {e}")
        return False

def test_dspy_optimizers():
    """Test DSPy optimizer parameter compatibility."""
    print("\n🔧 Testing DSPy Optimizer Parameters...")
    
    try:
        import dspy
        from dspy.teleprompt import BootstrapFewShot, MIPROv2
        from modules.metrics import get_comprehensive_metric
        
        # Test metric function
        def dummy_metric(example, pred, trace=None):
            """Dummy metric for testing optimizer initialization."""
            return 0.5
        
        # Test BootstrapFewShot with correct parameters
        print("📚 Testing BootstrapFewShot parameters...")
        try:
            bootstrap_optimizer = BootstrapFewShot(
                metric=dummy_metric,
                max_bootstrapped_demos=2,  # Small number for test
                max_labeled_demos=2,
                metric_threshold=0.5
                # num_candidate_programs=5,  # This parameter should NOT exist
            )
            print("✅ BootstrapFewShot initialized successfully")
        except Exception as e:
            print(f"❌ BootstrapFewShot failed: {e}")
            return False
        
        # Test MIPROv2 with correct parameters
        print("🧠 Testing MIPROv2 parameters...")
        try:
            mipro_optimizer = MIPROv2(
                metric=dummy_metric,
                auto="light",
                num_candidates=5,  # Should be num_candidates, NOT num_trials
                init_temperature=0.5,
                verbose=False,
                track_stats=True
            )
            print("✅ MIPROv2 initialized successfully")
        except Exception as e:
            print(f"❌ MIPROv2 failed: {e}")
            return False
        
        print("✅ All DSPy optimizer parameters are correct")
        return True
        
    except Exception as e:
        print(f"❌ DSPy optimizer test failed: {e}")
        return False

def test_multi_llm_configuration():
    """Test multi-LLM configuration for different components."""
    print("\n🤖 Testing Multi-LLM Configuration...")
    
    try:
        from agent import DSPyRAGChatAgent, _DSPyRAGProgram, create_llm_for_component
        from modules.utils import build_retriever
        import mlflow
        import tempfile
        import yaml
        
        # Create config with different LLMs for each component
        multi_llm_config = {
            "llm_config": {
                "endpoint": "databricks/databricks-meta-llama-3-1-8b-instruct",
                "max_tokens": 2500,
                "temperature": 0.01
            },
            "llm_endpoints": {
                "query_rewriter": {
                    "endpoint": "databricks/databricks-meta-llama-3-1-8b-instruct",
                    "max_tokens": 150,
                    "temperature": 0.3
                },
                "response_generator": {
                    "endpoint": "databricks/databricks-meta-llama-3-1-70b-instruct",
                    "max_tokens": 2500,
                    "temperature": 0.01
                },
                "optimization_judge": {
                    "endpoint": "databricks/databricks-claude-3-7-sonnet",
                    "max_tokens": 1000,
                    "temperature": 0.0
                }
            },
            "vector_search": {
                "index_fullname": "users.alex_miller.wikipedia_chunks_index",
                "text_column_name": "chunk",
                "docs_id_column_name": "id",
                "columns": ["id", "title", "chunk_id"],
                "top_k": 3
            },
            "agent_config": {
                "use_query_rewriter": True,
                "use_optimized": False,
                "enable_tracing": False
            }
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(multi_llm_config, f)
            config_path = f.name
        
        try:
            # Load config and test LLM creation
            config = mlflow.models.ModelConfig(development_config=config_path)
            
            # First, we need to update the global model_config for create_llm_for_component to work
            import agent
            agent.model_config = config
            
            # Test creating LLMs for different components
            query_rewriter_lm = create_llm_for_component("query_rewriter")
            response_generator_lm = create_llm_for_component("response_generator")
            optimization_judge_lm = create_llm_for_component("optimization_judge")
            
            print(f"✅ Query Rewriter LLM: {query_rewriter_lm.model}")
            print(f"✅ Response Generator LLM: {response_generator_lm.model}")
            print(f"✅ Optimization Judge LLM: {optimization_judge_lm.model}")
            
            # Verify different endpoints are used
            assert query_rewriter_lm.model == "databricks/databricks-meta-llama-3-1-8b-instruct"
            assert response_generator_lm.model == "databricks/databricks-meta-llama-3-1-70b-instruct"
            assert optimization_judge_lm.model == "databricks/databricks-claude-3-7-sonnet"
            
            # Test creating a program with multi-LLM config
            retriever = build_retriever(config)
            program = _DSPyRAGProgram(retriever, config)
            
            # Verify the program has different LLMs
            print(f"✅ Program query_rewriter_lm: {program.query_rewriter_lm.model}")
            print(f"✅ Program response_generator_lm: {program.response_generator_lm.model}")
            
            # Test with a simple question
            test_response = program("What is MLflow?")
            print(f"✅ Multi-LLM program executed successfully")
            
            return True
            
        finally:
            # Cleanup
            import os
            os.unlink(config_path)
        
    except Exception as e:
        print(f"❌ Multi-LLM configuration test failed: {e}")
        return False

def test_retrieval_validation():
    """Test retrieval context validation and debugging features."""
    print("\\n🔍 Testing Retrieval Context Validation...")
    
    try:
        from agent import _DSPyRAGProgram
        from modules.utils import build_retriever
        import mlflow
        import sys
        from io import StringIO
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        config = mlflow.models.ModelConfig(development_config=str(config_path))
        
        # Build retriever and program
        retriever = build_retriever(config)
        program = _DSPyRAGProgram(retriever, config)
        
        # Capture output for analysis
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # Test with a query that should return results
            test_query = "What is machine learning?"
            result = program(test_query)
            
            # Restore stdout
            sys.stdout = old_stdout
            output = captured_output.getvalue()
            
            # Analyze the captured debug output
            lines = output.split('\\n')
            retrieval_info = {}
            
            # Parse debug information
            for line in lines:
                if "🔍 Retrieved" in line:
                    # Extract number of documents
                    try:
                        doc_count = int(line.split("Retrieved ")[1].split(" documents")[0])
                        retrieval_info["doc_count"] = doc_count
                    except:
                        retrieval_info["doc_count"] = 0
                        
                elif "📄 Sample document type:" in line:
                    retrieval_info["doc_type"] = line.split("type: ")[1].strip()
                    
                elif "📋 Available fields:" in line or "📋 Available keys:" in line:
                    fields_str = line.split(": ")[1].strip()
                    retrieval_info["available_fields"] = fields_str
                    
                elif "📄 Passage" in line and "extracted via" in line:
                    if "passage_extractions" not in retrieval_info:
                        retrieval_info["passage_extractions"] = []
                    retrieval_info["passage_extractions"].append(line.strip())
                    
                elif "📊 Total context length:" in line:
                    try:
                        length = int(line.split("length: ")[1].split(" characters")[0])
                        retrieval_info["total_length"] = length
                    except:
                        retrieval_info["total_length"] = 0
                        
                elif "⚠️  Content extraction issues:" in line:
                    retrieval_info["extraction_issues"] = line.split("issues: ")[1].strip()
            
            # Print analysis results
            print(f"📊 Retrieval Analysis Results:")
            print(f"  - Documents Retrieved: {retrieval_info.get('doc_count', 'Unknown')}")
            print(f"  - Document Type: {retrieval_info.get('doc_type', 'Unknown')}")
            print(f"  - Available Fields: {retrieval_info.get('available_fields', 'Unknown')}")
            print(f"  - Total Context Length: {retrieval_info.get('total_length', 'Unknown')} chars")
            
            if "passage_extractions" in retrieval_info:
                print(f"  - Sample Extractions:")
                for extraction in retrieval_info["passage_extractions"][:2]:
                    print(f"    {extraction}")
            
            if "extraction_issues" in retrieval_info:
                print(f"  - ⚠️  Issues Found: {retrieval_info['extraction_issues']}")
            
            # Validation checks
            validation_results = []
            
            # Check 1: Documents were retrieved
            if retrieval_info.get("doc_count", 0) > 0:
                validation_results.append("✅ Documents were retrieved")
            else:
                validation_results.append("❌ No documents retrieved")
                
            # Check 2: Meaningful content length
            if retrieval_info.get("total_length", 0) > 100:
                validation_results.append("✅ Substantial content retrieved")
            elif retrieval_info.get("total_length", 0) > 50:
                validation_results.append("⚠️ Limited content retrieved")
            else:
                validation_results.append("❌ Very little content retrieved")
                
            # Check 3: No extraction issues
            if "extraction_issues" not in retrieval_info:
                validation_results.append("✅ No content extraction issues")
            else:
                validation_results.append(f"⚠️ Content extraction issues detected")
                
            # Check 4: Response quality
            if result.response and len(result.response.strip()) > 50:
                validation_results.append("✅ Generated meaningful response")
            else:
                validation_results.append("❌ Poor or empty response generated")
            
            print(f"\\n🔍 Validation Results:")
            for validation in validation_results:
                print(f"  {validation}")
            
            # Overall assessment
            issues = sum(1 for v in validation_results if v.startswith("❌"))
            warnings = sum(1 for v in validation_results if v.startswith("⚠️"))
            
            if issues == 0 and warnings <= 1:
                print("✅ Overall: Retrieval system working well")
                return True
            elif issues <= 1:
                print("⚠️ Overall: Retrieval system has some issues but is functional")
                return True
            else:
                print("❌ Overall: Significant retrieval system issues detected")
                return False
            
        finally:
            sys.stdout = old_stdout
        
    except Exception as e:
        print(f"❌ Retrieval validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_test():
    """Interactive testing mode for rapid iteration."""
    print("\n💬 Interactive Testing Mode")
    print("Type 'quit' to exit, 'help' for commands")
    
    try:
        from agent import DSPyRAGChatAgent, _DSPyRAGProgram
        from mlflow.types.agent import ChatAgentMessage
        from modules.utils import build_retriever
        import mlflow
        
        # Setup
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        config = mlflow.models.ModelConfig(development_config=str(config_path))
        
        # Choose mode
        print("\nSelect mode:")
        print("1. ChatAgent (full MLflow interface)")
        print("2. Direct Program (for rapid iteration)")
        
        mode = input("Enter choice (1 or 2): ").strip()
        
        if mode == "1":
            agent = DSPyRAGChatAgent(config=config)
            test_func = lambda q: agent.predict([ChatAgentMessage(role="user", content=q)])
        else:
            retriever = build_retriever(config)
            program = _DSPyRAGProgram(retriever, config)
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
        test_metrics_module,
        test_training_examples_generation,
        test_optimization_config,
        test_dspy_optimizers,
        test_multi_llm_configuration,
        test_retrieval_validation,
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