"""
DSPy RAG Agent Optimizer

This module contains all optimization-related functionality for DSPy RAG agents,
including training data preparation, evaluation metrics, and multi-stage optimization workflows.
Extracted from the main script for better modularity and testability.
"""

import os
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any, Callable

import dspy
import mlflow
from dspy.evaluate import Evaluate
from dspy.teleprompt import MIPROv2, BootstrapFewShot

from .metrics import (
    citation_accuracy_bool, 
    semantic_f1_bool, 
    end_to_end_bool,
    get_comprehensive_metric
)


# Fast optimization configuration for 5-10 minute runs
FAST_OPTIMIZATION_CONFIG = {
    "strategy": "bootstrap_only",           # Fast bootstrap-only strategy
    "auto_level": "light",                  # Light optimization level
    "num_threads": 4,                       # Parallel processing
    "training_examples_limit": 20,          # Reduced dataset for speed
    "evaluation_examples_limit": 5,         # Minimal evaluation set
    "bootstrap_config": {
        "max_bootstrapped_demos": 2,        # Minimal demos for speed
        "max_labeled_demos": 1,             # Single labeled example
        "metric_threshold": 0.3             # Standard threshold
    },
    "miprov2_config": {
        "metric_threshold": 0.5,            # Higher threshold for faster convergence
        "init_temperature": 1.0,            # Standard temperature
        "verbose": True                     # Enable progress tracking
    }
}

# Production optimization configuration
PRODUCTION_OPTIMIZATION_CONFIG = {
    "strategy": "multi_stage",              # Full multi-stage strategy  
    "auto_level": "medium",                 # Medium optimization level
    "num_threads": 8,                       # More parallel processing
    "training_examples_limit": 100,         # Full dataset
    "evaluation_examples_limit": 20,        # Comprehensive evaluation
    "bootstrap_config": {
        "max_bootstrapped_demos": 4,        # More examples for better performance
        "max_labeled_demos": 2,             # Multiple labeled examples
        "metric_threshold": 0.5             # Higher quality threshold
    },
    "miprov2_config": {
        "metric_threshold": 0.7,            # Higher threshold for production
        "init_temperature": 0.8,            # Slightly lower temperature
        "verbose": True                     # Enable progress tracking
    }
}


def prepare_training_data(
    spark, 
    config: Dict[str, Any],
    uc_catalog: str,
    uc_schema: str,
    eval_dataset_name: str = None
) -> List[dspy.Example]:
    """
    Prepare training data for DSPy optimization.
    
    Args:
        spark: Spark session
        config: Optimization configuration dictionary
        uc_catalog: Unity Catalog name
        uc_schema: Unity Catalog schema name
        eval_dataset_name: Name of evaluation dataset table
        
    Returns:
        List[dspy.Example]: List of training examples
    """
    print("\n1️⃣ Preparing training data...")
    
    training_limit = config.get('training_examples_limit', 50)
    
    if eval_dataset_name:
        # Load from Delta table
        eval_dataset_table = f"{uc_catalog}.{uc_schema}.{eval_dataset_name}"
        print(f"Loading evaluation dataset from: {eval_dataset_table}")
        eval_df = spark.table(eval_dataset_table)
        eval_data = eval_df.collect()
        
        training_examples = [
            dspy.Example(
                request=row['inputs']['messages'][0]['content'],
                response=row["expectations"]["expected_facts"]
            ).with_inputs("request")
            for row in eval_data[:training_limit]  # Use configured limit for training
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
            ).with_inputs("request")
        ][:training_limit]  # Limit hardcoded examples too
    
    print(f"✅ Prepared {len(training_examples)} training examples")
    return training_examples


def setup_evaluation_metric(judge_lm) -> Callable:
    """
    Set up the evaluation metric for DSPy optimization using a dedicated judge LM.
    
    Args:
        judge_lm: The DSPy language model to use for evaluation (separate from generation LM)
        
    Returns:
        Callable: The configured evaluation metric function
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
        """Evaluate RAG response against expected facts using dedicated judge LM."""
        request = example.request
        expected_facts = example.response  # This contains the expected facts
        generated_response = prediction.response
        
        with dspy.context(lm=judge_lm):
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
    
    print("✅ RAG evaluation metric configured with dedicated judge LM")
    return rag_evaluation_metric


def run_bootstrap_optimization(
    program: dspy.Module, 
    training_examples: List[dspy.Example], 
    metric: Callable,
    config: Dict[str, Any],
    lm
) -> Tuple[Optional[dspy.Module], float, Dict[str, Any]]:
    """
    Run Bootstrap Few-Shot optimization stage.
    
    Args:
        program: Base DSPy program to optimize
        training_examples: Training data
        metric: Evaluation metric function
        config: Bootstrap configuration
        lm: Language model for generation
        
    Returns:
        Tuple of (optimized_program, score, metadata)
    """
    print("\n📚 Stage 1: Bootstrap optimization...")
    
    try:
        bootstrap_config = config.get("bootstrap_config", {})
        bootstrap_optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=bootstrap_config.get("max_bootstrapped_demos", 2),
            max_labeled_demos=bootstrap_config.get("max_labeled_demos", 1),
            metric_threshold=bootstrap_config.get("metric_threshold", 0.3)
        )
        
        with dspy.context(lm=lm):
            bootstrap_program = bootstrap_optimizer.compile(
                program, 
                trainset=training_examples
            )
        
        # Evaluate bootstrap performance
        evaluator = Evaluate(
            devset=training_examples[:config.get('evaluation_examples_limit', 10)],
            metric=metric,
            num_threads=1,
            display_progress=True
        )
        
        with dspy.context(lm=lm):
            bootstrap_score = evaluator(bootstrap_program)
        
        print(f"✅ Bootstrap optimization completed: {bootstrap_score:.3f}")
        
        metadata = {
            "stage": "bootstrap",
            "score": bootstrap_score,
            "config": bootstrap_config,
            "timestamp": datetime.now().isoformat()
        }
        
        return bootstrap_program, bootstrap_score, metadata
        
    except Exception as e:
        print(f"❌ Bootstrap optimization failed: {e}")
        return None, 0.0, {"stage": "bootstrap", "error": str(e)}


def run_miprov2_optimization(
    program: dspy.Module, 
    training_examples: List[dspy.Example], 
    metric: Callable,
    config: Dict[str, Any],
    lm
) -> Tuple[Optional[dspy.Module], float, Dict[str, Any]]:
    """
    Run MIPROv2 optimization stage.
    
    Args:
        program: DSPy program to optimize (potentially from bootstrap stage)
        training_examples: Training data
        metric: Evaluation metric function
        config: MIPROv2 configuration
        lm: Language model for generation
        
    Returns:
        Tuple of (optimized_program, score, metadata)
    """
    print("\n🧠 Stage 2: MIPROv2 optimization...")
    
    try:
        miprov2_config = config.get("miprov2_config", {})
        miprov2_optimizer = MIPROv2(
            metric=metric,
            metric_threshold=miprov2_config.get("metric_threshold", 0.5),
            init_temperature=miprov2_config.get("init_temperature", 1.0),
            verbose=miprov2_config.get("verbose", True)
        )
        
        with dspy.context(lm=lm):
            miprov2_program = miprov2_optimizer.compile(
                program,
                trainset=training_examples,
                valset=training_examples[:config.get('evaluation_examples_limit', 10)]
            )
        
        # Evaluate MIPROv2 performance
        evaluator = Evaluate(
            devset=training_examples[:config.get('evaluation_examples_limit', 10)],
            metric=metric,
            num_threads=1,
            display_progress=True
        )
        
        with dspy.context(lm=lm):
            miprov2_score = evaluator(miprov2_program)
        
        print(f"✅ MIPROv2 optimization completed: {miprov2_score:.3f}")
        
        metadata = {
            "stage": "miprov2",
            "score": miprov2_score,
            "config": miprov2_config,
            "timestamp": datetime.now().isoformat()
        }
        
        return miprov2_program, miprov2_score, metadata
        
    except Exception as e:
        print(f"❌ MIPROv2 optimization failed: {e}")
        return None, 0.0, {"stage": "miprov2", "error": str(e)}


def evaluate_program_performance(
    program: dspy.Module,
    examples: List[dspy.Example],
    metric: Callable,
    lm,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    Evaluate program performance using multiple metrics.
    
    Args:
        program: DSPy program to evaluate
        examples: Evaluation examples
        metric: Primary evaluation metric
        lm: Language model
        detailed: Whether to run detailed multi-metric analysis
        
    Returns:
        Dict containing evaluation results
    """
    print("📊 Evaluating program performance...")
    
    evaluator = Evaluate(
        devset=examples,
        metric=metric,
        num_threads=1,
        display_progress=True
    )
    
    with dspy.context(lm=lm):
        primary_score = evaluator(program)
    
    results = {
        "primary_score": primary_score,
        "evaluation_size": len(examples),
        "timestamp": datetime.now().isoformat()
    }
    
    if detailed:
        # Run additional metric analysis on subset
        print("📊 Running detailed multi-metric analysis...")
        detailed_scores = {}
        
        for i, example in enumerate(examples[:3]):  # Analyze first 3 examples
            with dspy.context(lm=lm):
                pred = program(example.request)
            
            # Citation accuracy
            citation_score = citation_accuracy_bool(example, pred)
            
            # Semantic F1  
            semantic_score = semantic_f1_bool(example, pred)
            
            # End-to-end score
            e2e_score = end_to_end_bool(example, pred)
            
            detailed_scores[f"example_{i+1}"] = {
                "citation": citation_score,
                "semantic": semantic_score,
                "end_to_end": e2e_score
            }
            
            print(f"  Example {i+1}: Citation={citation_score}, Semantic={semantic_score}, E2E={e2e_score}")
        
        results["detailed_analysis"] = detailed_scores
    
    return results


def save_optimization_artifacts(
    program: dspy.Module,
    metadata: Dict[str, Any], 
    script_dir: str,
    run_name: str = "optimized_program"
) -> str:
    """
    Save optimization artifacts to local directory and log metadata to MLflow.
    
    Args:
        program: Optimized DSPy program
        metadata: Optimization metadata
        script_dir: Directory to save artifacts
        run_name: Name for the optimization run
        
    Returns:
        str: Path to saved program
    """
    print("💾 Saving optimization artifacts...")
    
    # Create artifacts directory
    artifacts_dir = os.path.join(script_dir, "optimization_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save optimized program
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    program_path = os.path.join(artifacts_dir, f"{run_name}_{timestamp}.pkl")
    
    dspy.teleprompt.save_program(program, program_path)
    print(f"✅ Optimized program saved to: {program_path}")
    
    # Save metadata
    metadata_path = os.path.join(artifacts_dir, f"{run_name}_{timestamp}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    # Log to MLflow if in active run
    try:
        if mlflow.active_run():
            mlflow.log_artifact(program_path)
            mlflow.log_artifact(metadata_path)
            mlflow.log_metrics({
                f"{metadata.get('stage', 'optimization')}_score": metadata.get('score', 0.0)
            })
            print("✅ Artifacts logged to MLflow")
    except Exception as e:
        print(f"⚠️ Failed to log to MLflow: {e}")
    
    return program_path


def run_optimization_workflow(
    spark,
    lm,
    judge_lm, 
    base_program: dspy.Module,
    config: Dict[str, Any],
    uc_catalog: str,
    uc_schema: str,
    script_dir: str,
    eval_dataset_name: str = None,
    use_fast_config: bool = False
) -> Tuple[Optional[dspy.Module], Optional[str], Dict[str, Any]]:
    """
    Run the complete DSPy optimization workflow with multi-stage strategy.
    
    Args:
        spark: Spark session
        lm: The DSPy language model for generation
        judge_lm: The DSPy language model for evaluation
        base_program: The base DSPy program to optimize
        config: Base configuration
        uc_catalog: Unity Catalog name
        uc_schema: Unity Catalog schema
        script_dir: Script directory for saving artifacts
        eval_dataset_name: Name of evaluation dataset
        use_fast_config: Whether to use fast optimization for testing
        
    Returns:
        Tuple of (optimized_program, optimized_program_path, optimization_results)
    """
    print("🚀 Starting DSPy optimization process...")
    
    # Use fast config if requested, otherwise use provided config
    optimization_config = FAST_OPTIMIZATION_CONFIG if use_fast_config else config
    
    print(f"📋 Using {'FAST' if use_fast_config else 'STANDARD'} optimization configuration:")
    print(f"  - Strategy: {optimization_config.get('strategy')}")
    print(f"  - Training examples: {optimization_config.get('training_examples_limit')}")
    print(f"  - Evaluation examples: {optimization_config.get('evaluation_examples_limit')}")
    
    # SECTION 1: Prepare Training Data
    training_examples = prepare_training_data(
        spark, optimization_config, uc_catalog, uc_schema, eval_dataset_name
    )
    
    # SECTION 2: Setup Evaluation Metric  
    rag_evaluation_metric = setup_evaluation_metric(judge_lm)
    
    # SECTION 3: Evaluate Baseline
    print("\n3️⃣ Evaluating baseline performance...")
    baseline_results = evaluate_program_performance(
        base_program, 
        training_examples[:optimization_config.get('evaluation_examples_limit', 10)],
        rag_evaluation_metric,
        lm,
        detailed=not use_fast_config  # Skip detailed analysis in fast mode
    )
    baseline_score = baseline_results["primary_score"]
    print(f"📊 Baseline Score: {baseline_score:.3f}")
    
    # SECTION 4: Multi-Stage Optimization Strategy
    print("\n4️⃣ Running optimization strategy...")
    
    strategy = optimization_config.get("strategy", "multi_stage")
    optimized_program = base_program
    best_score = baseline_score
    optimization_history = []
    
    if strategy == "bootstrap_only" or strategy == "multi_stage":
        # Stage 1: Bootstrap optimization
        bootstrap_program, bootstrap_score, bootstrap_metadata = run_bootstrap_optimization(
            base_program, training_examples, rag_evaluation_metric, optimization_config, lm
        )
        
        if bootstrap_program and bootstrap_score > best_score:
            optimized_program = bootstrap_program
            best_score = bootstrap_score
            optimization_history.append(bootstrap_metadata)
            print(f"🎯 Bootstrap improved score: {baseline_score:.3f} → {bootstrap_score:.3f}")
        else:
            print("📊 Bootstrap did not improve performance, keeping baseline")
    
    if strategy == "multi_stage" and not use_fast_config:
        # Stage 2: MIPROv2 optimization (skip in fast mode)
        current_program = optimized_program if optimization_history else base_program
        miprov2_program, miprov2_score, miprov2_metadata = run_miprov2_optimization(
            current_program, training_examples, rag_evaluation_metric, optimization_config, lm
        )
        
        if miprov2_program and miprov2_score > best_score:
            optimized_program = miprov2_program
            best_score = miprov2_score
            optimization_history.append(miprov2_metadata)
            print(f"🎯 MIPROv2 improved score: {best_score:.3f} → {miprov2_score:.3f}")
        else:
            print("📊 MIPROv2 did not improve performance, keeping current best")
    
    # SECTION 5: Save Optimization Results
    final_results = {
        "baseline_score": baseline_score,
        "final_score": best_score,
        "improvement": best_score - baseline_score,
        "improvement_percent": ((best_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0,
        "strategy": strategy,
        "optimization_history": optimization_history,
        "config": optimization_config,
        "fast_mode": use_fast_config,
        "timestamp": datetime.now().isoformat()
    }
    
    optimized_program_path = None
    if optimized_program != base_program:
        # Save optimized program
        optimized_program_path = save_optimization_artifacts(
            optimized_program, final_results, script_dir
        )
        
        print(f"\n🎉 Optimization completed successfully!")
        print(f"📊 Final Results:")
        print(f"  - Baseline: {baseline_score:.3f}")
        print(f"  - Optimized: {best_score:.3f}")
        print(f"  - Improvement: {final_results['improvement']:.3f} ({final_results['improvement_percent']:.1f}%)")
        print(f"  - Strategy: {strategy}")
        print(f"  - Optimized program saved to: {optimized_program_path}")
    else:
        print(f"\n📊 Optimization completed - no improvement found")
        print(f"  - Baseline score: {baseline_score:.3f}")
        print(f"  - Using baseline program for deployment")
    
    return optimized_program, optimized_program_path, final_results


def get_optimization_config(mode: str = "fast") -> Dict[str, Any]:
    """
    Get optimization configuration for different modes.
    
    Args:
        mode: "fast" for 5-10 minute runs, "production" for full optimization
        
    Returns:
        Dict containing optimization configuration
    """
    if mode == "fast":
        return FAST_OPTIMIZATION_CONFIG.copy()
    elif mode == "production":
        return PRODUCTION_OPTIMIZATION_CONFIG.copy()
    else:
        raise ValueError(f"Unknown optimization mode: {mode}. Use 'fast' or 'production'")