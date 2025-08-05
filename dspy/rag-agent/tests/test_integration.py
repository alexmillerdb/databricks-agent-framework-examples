#!/usr/bin/env python3
"""
Integration Test for Refactored DSPy RAG Agent

This script tests the integration between the new modular components
to ensure the refactoring maintains functionality.
"""

import os
import sys
from datetime import datetime

def test_module_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing module imports...")
    
    try:
        # Test core imports
        import mlflow
        import dspy
        print("  ✅ Core libraries (mlflow, dspy)")
        
        # Add parent directory to path for agent import
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, parent_dir)
        
        # Test our modular components
        from modules.utils import (
            print_workflow_configuration, 
            build_retriever,
            format_duration,
            validate_config_completeness,
            get_config_value
        )
        print("  ✅ Utils module")
        
        from modules.optimizer import (
            get_optimization_config,
            prepare_training_data,
            setup_evaluation_metric
        )
        print("  ✅ Optimizer module")
        
        from modules.deploy import (
            create_final_agent,
            log_model_to_mlflow,
            full_deployment_workflow
        )
        print("  ✅ Deploy module")
        
        from agent import DSPyRAGChatAgent, _DSPyRAGProgram
        print("  ✅ Agent module")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_configuration_functions():
    """Test configuration handling functions."""
    print("\n🔧 Testing configuration functions...")
    
    try:
        from modules.optimizer import get_optimization_config
        from modules.utils import validate_config_completeness, format_duration
        
        # Test optimization configs
        fast_config = get_optimization_config("fast")
        prod_config = get_optimization_config("production")
        
        assert fast_config["strategy"] == "bootstrap_only"
        assert prod_config["strategy"] == "multi_stage"
        assert fast_config["training_examples_limit"] < prod_config["training_examples_limit"]
        print("  ✅ Optimization configuration functions")
        
        # Test config validation
        test_config = {
            "llm_config": {
                "endpoint": "test",
                "max_tokens": 1000,
                "temperature": 0.1
            },
            "vector_search": {
                "index_fullname": "test.index",
                "top_k": 5
            }
        }
        
        is_valid, missing = validate_config_completeness(test_config)
        assert is_valid, f"Config validation failed: {missing}"
        print("  ✅ Configuration validation functions")
        
        # Test utility functions
        duration_str = format_duration(3661)  # 1 hour, 1 minute, 1 second
        assert "1.0h" in duration_str
        print("  ✅ Utility functions")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def test_workflow_configuration():
    """Test workflow configuration printing."""
    print("\n📋 Testing workflow configuration...")
    
    try:
        from modules.utils import print_workflow_configuration
        from modules.optimizer import get_optimization_config
        
        optimization_config = get_optimization_config("fast")
        
        # This should not raise an exception
        print_workflow_configuration(
            optimize_agent=True,
            deploy_model=False,
            config_file="config.yaml",
            eval_dataset_table="test.table",
            optimization_config=optimization_config
        )
        
        print("  ✅ Workflow configuration display")
        return True
        
    except Exception as e:
        print(f"  ❌ Workflow configuration test failed: {e}")
        return False


def test_file_line_counts():
    """Test that the refactoring achieved the target line reduction."""
    print("\n📏 Testing file line counts...")
    
    try:
        # Get parent directory since tests are now in subdirectory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Check main script line count
        main_script = os.path.join(script_dir, "03-build-dspy-rag-agent.py")
        with open(main_script, 'r') as f:
            main_lines = len(f.readlines())
        
        print(f"  📄 Main script: {main_lines} lines")
        
        # Target was ~300 lines, allow up to 450 for buffer
        assert main_lines <= 450, f"Main script too long: {main_lines} lines"
        print("  ✅ Main script size target achieved")
        
        # Check that new modules exist and have reasonable size
        modules_to_check = [
            ("modules/optimizer.py", "optimizer.py"),
            ("modules/deploy.py", "deploy.py"),
            ("modules/utils.py", "utils.py"),
            ("modules/metrics.py", "metrics.py"),
            ("agent.py", "agent.py")
        ]
        total_module_lines = 0
        
        for module_path, module_name in modules_to_check:
            module_full_path = os.path.join(script_dir, module_path)
            if os.path.exists(module_full_path):
                with open(module_full_path, 'r') as f:
                    lines = len(f.readlines())
                total_module_lines += lines
                print(f"  📄 {module_name}: {lines} lines")
        
        print(f"  📊 Total modular lines: {total_module_lines}")
        print(f"  📊 Original script was ~1126 lines, now modularized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Line count test failed: {e}")
        return False


def test_fast_optimization_config():
    """Test that fast optimization configuration is properly set up."""
    print("\n⚡ Testing fast optimization configuration...")
    
    try:
        from modules.optimizer import FAST_OPTIMIZATION_CONFIG, get_optimization_config
        
        fast_config = get_optimization_config("fast")
        
        # Check that fast config has expected properties
        assert fast_config["strategy"] == "bootstrap_only"
        assert fast_config["training_examples_limit"] <= 20
        assert fast_config["evaluation_examples_limit"] <= 10
        assert fast_config["bootstrap_config"]["max_bootstrapped_demos"] <= 3
        
        print("  ✅ Fast optimization config properly configured")
        print(f"    Strategy: {fast_config['strategy']}")
        print(f"    Training examples: {fast_config['training_examples_limit']}")
        print(f"    Evaluation examples: {fast_config['evaluation_examples_limit']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Fast optimization config test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 DSPy RAG Agent - Integration Tests")
    print("=" * 50)
    
    tests = [
        test_module_imports,
        test_configuration_functions,
        test_workflow_configuration,
        test_file_line_counts,
        test_fast_optimization_config
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("🏁 Integration Test Results")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All integration tests passed! Refactoring successful.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Review and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())