# Databricks notebook source
# MAGIC %md
# MAGIC # DSPy RAG Compilation & Optimization
# MAGIC 
# MAGIC This notebook compiles and optimizes the DSPy RAG program for better performance,
# MAGIC then saves the optimized version for deployment using MLflow ModelConfig.

# COMMAND ----------
%pip install -qqq --upgrade "mlflow[databricks]>=3.1" dspy-ai databricks-agents openai
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Configuration Setup
# MAGIC 
# MAGIC Load configuration using MLflow ModelConfig for parameterized optimization.

# COMMAND ----------
import os
import dspy
import mlflow
import json
from typing import List, Dict, Any
import pickle

# Load development configuration
config_path = "config.yaml"
model_config = mlflow.models.ModelConfig(development_config=config_path)

# Configuration from ModelConfig
catalog = "users"
schema = "alex_miller"
vector_search_index_name = "wikipedia_chunks_index"
UC_VS_INDEX_NAME = f"{catalog}.{schema}.{vector_search_index_name}"

# Set environment variables from config
llm_config = model_config.get("llm_config") or {}
os.environ["VS_INDEX_FULLNAME"] = UC_VS_INDEX_NAME
os.environ["DSPY_LLM_ENDPOINT"] = llm_config.get("endpoint", "databricks/databricks-claude-3-7-sonnet")

print(f"Using LLM endpoint: {os.environ['DSPY_LLM_ENDPOINT']}")
print(f"Using Vector Search index: {UC_VS_INDEX_NAME}")
print(f"Configuration loaded from: {config_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## DSPy and MLflow Configuration

# COMMAND ----------
# Configure DSPy with settings from config
lm = dspy.LM(
    model=os.environ["DSPY_LLM_ENDPOINT"],
    max_tokens=llm_config.get("max_tokens", 2500),
    temperature=llm_config.get("temperature", 0.01),
    top_p=llm_config.get("top_p", 0.95)
)
dspy.configure(lm=lm)

# MLflow configuration
mlflow.set_tracking_uri("databricks")
user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Set experiment based on config
mlflow_config = model_config.get("mlflow_config") or {}
experiment_name = f"{mlflow_config.get('experiment_name', 'dspy_rag_agent')}_optimization"
mlflow.set_experiment(f"/Users/{user_name}/{experiment_name}")

# Configure tracing based on config
if mlflow_config.get("enable_autolog", True):
    mlflow.dspy.autolog(log_traces_from_compile=True)

# Import the base RAG program
from agent import _DSPyRAGProgram
from utils import build_retriever

# COMMAND ----------
# MAGIC %md
# MAGIC ## Define Training Dataset

# COMMAND ----------
def create_training_dataset() -> List[dspy.Example]:
    """Create training examples for optimization."""
    training_data = [
        {"request": "Who is the son of Zeus?", "expected_response": "Hercules"},
        {"request": "Who is Zeus?", "expected_response": "A Greek god"},
        {
            "request": "What can you tell me about Greek mythology?",
            "expected_response": "Greek myth takes many forms, from religious myths of origin to folktales and legends of heroes",
        },
        {
            "request": "Who is Frederick Barbarossa?",
            "expected_response": "King of Germany in 1152 and Holy Roman Emperor in 1155",
        },
        {
            "request": "When was Frederick Barbarossa a king?",
            "expected_response": "In the year eleven hundred fifty two",
        },
        {
            "request": "Which kingdom did Frederick Barbarossa rule?",
            "expected_response": "Kingdom of Germany",
        },
        {
            "request": "Who is Tom McNab?",
            "expected_response": "Tom McNab has been national champion for triple jump five times and is the author of 'The Complete Book of Track and Field'",
        },
        {
            "request": "Who wrote 'The Complete Book of Track and Field'?",
            "expected_response": "Tom McNab",
        },
    ]
    
    return [
        dspy.Example(**item).with_inputs("request") 
        for item in training_data
    ]

def create_training_dataset_from_delta(
    catalog: str = "users",
    schema: str = "alex_miller", 
    table_name: str = "rag_training_data",
    question_col: str = "request",
    answer_col: str = "expected_response",
    limit: int = None
) -> List[dspy.Example]:
    """Create training examples for optimization from a Delta table in UC."""
    
    # Construct the full table name
    full_table_name = f"{catalog}.{schema}.{table_name}"
    
    try:
        # Read from Delta table using Spark
        df = spark.table(full_table_name)
        
        # Optionally limit the number of rows for faster training
        if limit:
            df = df.limit(limit)
        
        # Convert to Pandas for easier manipulation
        pandas_df = df.select(question_col, answer_col).toPandas()
        
        # Convert to the format expected by DSPy
        training_data = []
        for _, row in pandas_df.iterrows():
            training_data.append({
                "request": row[question_col],
                "expected_response": row[answer_col]
            })
        
        print(f"Loaded {len(training_data)} training examples from {full_table_name}")
        
        # Convert to DSPy Examples
        return [
            dspy.Example(**item).with_inputs("request") 
            for item in training_data
        ]
        
    except Exception as e:
        print(f"Error reading from Delta table {full_table_name}: {e}")
        print("Falling back to hardcoded examples...")
        
        # Fallback to your current hardcoded examples
        return create_training_dataset()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Define Evaluation Metric

# COMMAND ----------
class ResponseAssessment(dspy.Signature):
    """Assess if a response correctly answers a question."""
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Answer with 'correct' or 'incorrect' only")

def rag_evaluation_metric(gold, pred, trace=None):
    """Custom metric for evaluating RAG responses."""
    request = gold.request
    expected = gold.expected_response
    actual = pred.response
    
    prompt = f"Rate how well the response '{actual}' answers the question '{request}' compared to the expected answer '{expected}'. The response should contain similar information or facts. Respond with only 'correct' or 'incorrect'."
    
    try:
        judge = dspy.Predict(ResponseAssessment)(
            assessed_text=actual, 
            assessment_question=prompt
        )
        
        # More robust evaluation
        response = judge.assessment_answer.lower().strip()
        is_correct = "correct" in response and "incorrect" not in response
        
        # Always return float for consistency
        return 1.0 if is_correct else 0.0
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0.0

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test Metric and Base Program

# COMMAND ----------
def test_metric_and_base_program():
    """Test the metric and base program before optimization."""
    print("Testing metric and base program...")
    
    # Create base program and dataset using config
    retriever = build_retriever(model_config)
    base_program = _DSPyRAGProgram(retriever)
    trainset = create_training_dataset()
    
    print(f"Training dataset size: {len(trainset)}")
    
    # Test metric on a few examples
    print("\nTesting metric on sample examples:")
    for i, example in enumerate(trainset[:3]):
        try:
            pred = base_program(example.request)
            score = rag_evaluation_metric(example, pred)
            print(f"\nExample {i+1}:")
            print(f"Question: {example.request}")
            print(f"Expected: {example.expected_response}")
            print(f"Actual: {pred.response}")
            print(f"Score: {score}")
        except Exception as e:
            print(f"Error testing example {i+1}: {e}")
    
    return base_program, trainset

# Test before optimization
base_program, trainset = test_metric_and_base_program()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compile & Optimize the RAG Program

# COMMAND ----------
def compile_rag_program(create_dataset_function=create_training_dataset):
    """Compile and optimize the DSPy RAG program using configuration."""
    
    # Create base program with config
    retriever = build_retriever(model_config)
    base_program = _DSPyRAGProgram(retriever)
    
    # Create training dataset
    if create_dataset_function == 'create_training_dataset':
        trainset = create_training_dataset()
    elif create_dataset_function == 'create_training_dataset_from_delta':
        trainset = create_training_dataset_from_delta()
    else:
        raise ValueError(f"Invalid create_dataset_function: {create_dataset_function}")
    
    # Evaluate base program first
    from dspy.evaluate import Evaluate
    evaluator = Evaluate(metric=rag_evaluation_metric, devset=trainset[:10])  # Use subset for faster evaluation
    
    print("Evaluating base program...")
    base_score = evaluator(base_program)
    print(f"Base program score: {base_score}")
    
    # Get optimization settings from config
    agent_config = model_config.get("agent_config") or {}
    max_iterations = agent_config.get("max_iterations", 20)
    
    # Initialize optimizer with verbose logging
    optimizer = dspy.MIPROv2(
        metric=rag_evaluation_metric,
        auto="medium",  # Can be "light", "medium", or "heavy"
        num_threads=8,  # Reduced for stability
        verbose=True,
        track_stats=True
    )
    
    # Compile the program
    print("\nStarting compilation...")
    try:
        optimized_program = optimizer.compile(
            base_program, 
            trainset=trainset,
            num_trials=max_iterations,
            max_bootstrapped_demos=4,
            max_labeled_demos=4
        )
        print("Compilation completed!")
        
        # Evaluate optimized program
        print("Evaluating optimized program...")
        optimized_score = evaluator(optimized_program)
        print(f"Optimized program score: {optimized_score}")
        print(f"Improvement: {optimized_score - base_score}")
        
        return optimized_program, trainset, base_score, optimized_score
        
    except Exception as e:
        print(f"Error during compilation: {e}")
        print("Falling back to BootstrapFewShot optimizer...")
        
        # Fallback to simpler optimizer
        from dspy.teleprompt import BootstrapFewShotWithRandomSearch
        
        fallback_optimizer = BootstrapFewShotWithRandomSearch(
            metric=rag_evaluation_metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_candidate_programs=10,
            num_threads=4
        )
        
        optimized_program = fallback_optimizer.compile(base_program, trainset=trainset)
        optimized_score = evaluator(optimized_program)
        print(f"Fallback optimized program score: {optimized_score}")
        print(f"Improvement: {optimized_score - base_score}")
        
        return optimized_program, trainset, base_score, optimized_score

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run Compilation & Save Results

# COMMAND ----------
with mlflow.start_run(run_name="DSPy_RAG_Compilation"):
    # Compile the program
    optimized_program, trainset, base_score, optimized_score = compile_rag_program()
    
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
        model_path = "optimized_rag_program.json"
        optimized_program.save(model_path)
        
        # Log the optimized program as an artifact
        mlflow.log_artifact(model_path, "optimized_program")
        print("Optimized program saved successfully!")
        
    except Exception as e:
        print(f"Error saving program: {e}")
    
    # Save program metadata with config info
    program_state = {
        "program_type": type(optimized_program).__name__,
        "base_score": float(base_score),
        "optimized_score": float(optimized_score),
        "improvement": float(optimized_score - base_score),
        "config": {
            "optimizer": "MIPROv2",
            "training_examples": len(trainset),
            "optimization_level": "medium",
            "llm_config": llm_config,
            "agent_config": model_config.get("agent_config") or {},
            "config_file": config_path
        }
    }
    
    with open("program_metadata.json", "w") as f:
        json.dump(program_state, f, indent=2)
    
    mlflow.log_artifact("program_metadata.json", "optimized_program")
    
    # Log the model config as an artifact
    mlflow.log_dict(model_config.to_dict(), "model_config.json")
    
    # Log metrics and parameters
    mlflow.log_param("training_examples", len(trainset))
    mlflow.log_param("optimizer", "MIPROv2")
    mlflow.log_param("optimization_level", "medium")
    mlflow.log_param("config_file", config_path)
    mlflow.log_param("llm_endpoint", llm_config.get("endpoint"))
    mlflow.log_param("max_tokens", llm_config.get("max_tokens"))
    mlflow.log_param("temperature", llm_config.get("temperature"))
    mlflow.log_metric("base_score", base_score)
    mlflow.log_metric("optimized_score", optimized_score)
    mlflow.log_metric("improvement", optimized_score - base_score)
    
    print(f"\nOptimization Results:")
    print(f"Base Score: {base_score:.3f}")
    print(f"Optimized Score: {optimized_score:.3f}")
    print(f"Improvement: {optimized_score - base_score:.3f}")
    print(f"Configuration: {config_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Inspect Optimized Program

# COMMAND ----------
# Inspect the history to see how the prompts changed
try:
    # Inspect the history to see how the prompts changed
    print("Inspecting optimization history...")
    if hasattr(optimized_program, 'response_generator'):
        optimized_program.response_generator.inspect_history()
    else:
        print("No response_generator found or history not available")
        
    # Print the final optimized prompts
    print("\nFinal optimized program structure:")
    for name, module in optimized_program.named_predictors():
        print(f"Module: {name}")
        if hasattr(module, 'signature'):
            print(f"  Signature: {module.signature}")
        if hasattr(module, 'extended_signature'):
            print(f"  Extended Signature: {module.extended_signature}")
        print()
        
except Exception as e:
    print(f"Error inspecting program: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Additional Evaluation

# COMMAND ----------
def detailed_evaluation():
    """Perform detailed evaluation of the optimized program."""
    
    # Create a separate test set for final evaluation
    test_examples = [
        {"request": "What is artificial intelligence?", "expected_response": "AI is the simulation of human intelligence in machines"},
        {"request": "Who discovered penicillin?", "expected_response": "Alexander Fleming discovered penicillin"},
        {"request": "What is the theory of relativity?", "expected_response": "Einstein's theory describing the relationship between space and time"},
    ]
    
    test_set = [dspy.Example(**item).with_inputs("request") for item in test_examples]
    
    print("Detailed evaluation on test set:")
    total_score = 0
    
    for i, example in enumerate(test_set):
        try:
            pred = optimized_program(example.request)
            score = rag_evaluation_metric(example, pred)
            total_score += score
            
            print(f"\nTest {i+1}:")
            print(f"Question: {example.request}")
            print(f"Expected: {example.expected_response}")
            print(f"Actual: {pred.response}")
            print(f"Score: {score}")
            
        except Exception as e:
            print(f"Error in test {i+1}: {e}")
    
    avg_score = total_score / len(test_set) if test_set else 0
    print(f"\nAverage test score: {avg_score:.3f}")
    
    return avg_score

# Run detailed evaluation
test_score = detailed_evaluation()
mlflow.log_metric("test_score", test_score)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC * Use the optimized program in the deployment notebook (04-deploy-optimized-agent.py)
# MAGIC * Consider using `config_optimized.yaml` for production-optimized settings
# MAGIC * The optimized program artifacts are saved in MLflow for reproducibility