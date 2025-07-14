# Databricks notebook source
# MAGIC %md
# MAGIC # Part 2: Create and optimize a DSPy program for RAG
# MAGIC
# MAGIC This notebook shows how to:
# MAGIC * Create a basic RAG DSPy program.
# MAGIC * Run the DSPy program from a notebook.
# MAGIC * Optimize prompts using DSPy `BootstrapFewShot` optimizer.
# MAGIC * Run the optimized DSPy program.
# MAGIC
# MAGIC This notebook is part 2 of 2 notebooks for creating a DSPy program for RAG.
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC This notebook assumes:
# MAGIC * You have completed and run the [Part 1: Prepare data and vector search index for a RAG DSPy program](https://docs.databricks.com/_extras/notebooks/source/generative-ai/dspy/dspy-data-preparation.html) notebook.
# MAGIC * You have specified the following information in the notebook widgets:
# MAGIC   * `vs_index`: Databricks Vector Search index to be used in the RAG program.
# MAGIC   * `source_catalog`: UC catalog of the schema where the index is located.
# MAGIC   * `source_schema`: UC schema containing the Vector Search index.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Install dependencies

# COMMAND ----------

# %pip install -qqqq dspy-ai>=2.5.0 openai<2 databricks-agents>=0.5.0 mlflow>=2.1.6.0
%pip install --upgrade "mlflow[databricks]>=3.1" dspy-ai databricks-agents openai
dbutils.library.restartPython()

# COMMAND ----------

import dspy

dspy.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC ###Define notebook widgets
# MAGIC

# COMMAND ----------

dbutils.widgets.removeAll()
format_widget_name = lambda x: x.replace('_', ' ').title()

widget_defaultss = {
    "source_catalog": "users", # PLEASE ENTER YOUR CATALOG
    "source_schema": "alex_miller", # PLEASE ENTER YOUR SCHEMA
    "vs_index": "wikipedia_chunks_index", # PLEASE ENTER YOUR VECTOR SEARCH INDEX
}
for k, v in widget_defaultss.items():
    dbutils.widgets.text(k, v, format_widget_name(k))


# COMMAND ----------

# MAGIC %md
# MAGIC ###Define configurations
# MAGIC
# MAGIC The following example shows how to obtain a personal access token from the session using the specified notebook widget values. However, this method is not recommended for production; instead use a Databricks secret ([AWS](https://docs.databricks.com/en/security/secrets/index.html)|[ Azure](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/))

# COMMAND ----------

from dbruntime.databricks_repl_context import get_context

print("CONFIGURATIONS")
config = {}
for k in widget_defaultss.keys():
    config[k] = dbutils.widgets.get(k)
    assert config[k].strip() != "", f"Please provide a valid {format_widget_name(k)}"
    print(f"- config['{k}']= '{config[k]}'")

config[
    "vs_index_fullname"
] = f"{config['source_catalog']}.{config['source_schema']}.{config['vs_index']}"

print(f"- config['vs_index_fullname']= '{config['vs_index_fullname']}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Define the DSPy program
# MAGIC
# MAGIC A DSPy program consists of a Python class inherited from `dspy.Module` that implements the `forward()` method,  which runs the following steps:
# MAGIC   - Query a Databricks Vector Search index to retrieve document chunks (`context`) related to the `request`.
# MAGIC   - Generate an `response` by sending the `context` containing the document chunks and the `request` to an LLM.
# MAGIC
# MAGIC The `__init__` function initializes the resources the `forward` function uses. In this example, the resources are: 
# MAGIC   - `retrieve`: Databricks Vector Search retriever 
# MAGIC   - `lm`: Databricks Foundation Model pay-per-token ``Llama3-1-70B-instruct``
# MAGIC   - `response_generator`: The prediction technique, in this case [DSPy.predict](https://dspy-docs.vercel.app/api/modules/Predict), that uses an LLM to process retrieved documents and instructions to generate a response. Additional prediction techniques include [dspy.ChainOfThought](https://dspy-docs.vercel.app/api/modules/ChainOfThought) and [dspy.ReAct](https://dspy-docs.vercel.app/api/modules/ReAct).

# COMMAND ----------

import os

os.getcwd()

# COMMAND ----------

import dspy
# from dspy import Databricks
from dspy.retrieve.databricks_rm import DatabricksRM
import mlflow
import os


# Enabling tracing for DSPy
mlflow.dspy.autolog()

# Set up MLflow tracking to Databricks
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/alex.miller@databricks.com/dspy")

class RAG(dspy.Module):
    def __init__(self, num_passages=5):
        super().__init__()

        # Define the retriever that fetches relevant documents from the Databricks Vector Search index
        self.retriever = DatabricksRM(
            databricks_index_name=config["vs_index_fullname"],
            text_column_name="chunk",
            docs_id_column_name="id",
            columns=["id", "title", "chunk_id"],
            k=5,
        )
        # Define the language model that will be used for response generation
        self.lm = dspy.LM("databricks/databricks-claude-3-7-sonnet")

        # Define the program signature
        # The response generator will be provided with a "context" and a "request",
        # and will return a "response"
        signature = "context, request -> response"

        # Define response generator
        # self.response_generator = dspy.Predict(signature)
        self.response_generator = dspy.ChainOfThought(signature)

    def forward(self, request):

        # Obtain context by executing a Databricks Vector Search query
        retrieved_context = self.retriever(request)

        # Generate a response using the language model defined in the __init__ method
        with dspy.context(lm=self.lm):
            response = self.response_generator(
                context=retrieved_context.docs, request=request
            ).response

        return dspy.Prediction(response=response)



# COMMAND ----------

# MAGIC %md
# MAGIC ###Run the program
# MAGIC
# MAGIC To run the DSPy program, instantiate it and pass in the `request`.

# COMMAND ----------

# Instantiating DSPy program
rag = RAG()

# Running a query
result = rag("Who is Zeus?")

# Printing response
print(result.response)

# COMMAND ----------

# MAGIC %md
# MAGIC Not bad for such a simple program!!
# MAGIC
# MAGIC Try another query:

# COMMAND ----------

# Running another query
result = rag("Who is Augustus?")

# Printing response
print(result.response)

# COMMAND ----------

# MAGIC %md
# MAGIC This response is unxpected, since the program should have responded with `Yes`. When this happens, you can inspect the prompt generated by DSPy.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Inspecting generated prompt

# COMMAND ----------

rag.lm.inspect_history()

# COMMAND ----------

# MAGIC %md
# MAGIC You can see it is a simple prompt with minimal instructions.  Try optimizing it by providing few-shot examples.  DSPy selects which few-shot examples are most effective based on an evaluation criteria.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Optimizing prompts

# COMMAND ----------

# MAGIC %md
# MAGIC ####Define training set
# MAGIC
# MAGIC First,  define eight examples of `request` and `expected_response` pairs.

# COMMAND ----------

train_set = [
    # Defining a list of DSPy examples taking "request" as the input
    dspy.Example(**item).with_inputs("request")
    for item in [
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
]

# COMMAND ----------

train_set

# COMMAND ----------

# MAGIC %md ### Tracing during evaluation:
# MAGIC
# MAGIC Evaluating DSPy models is an important step in the development of AI systems. MLflow Tracing can help you track the performance of your programs after the evaluation, by providing detailed information about the execution of your programs for each input.
# MAGIC
# MAGIC When MLflow auto-tracing is enabled for DSPy, traces will be automatically generated when you execute DSPy's built-in evaluation suites. The following example demonstrates how to run evaluation and review traces in MLflow:

# COMMAND ----------

import dspy
from dspy.evaluate.metrics import answer_exact_match
import mlflow, os

dspy.settings.configure(
    lm=dspy.LM(model="databricks/databricks-claude-3-7-sonnet")
)

mlflow.dspy.autolog(log_traces_from_eval=True)

# ---------- 1. Prepare the dataset ----------
train_set = [
    dspy.Example(question="Who is the son of Zeus?",  answer="Hercules").with_inputs("question"),
    dspy.Example(question="Who is Zeus?",             answer="A Greek god").with_inputs("question"),
]

# ---------- 2. Define (or import) the program to evaluate ----------
class Counter(dspy.Signature):
    question = dspy.InputField()
    answer   = dspy.OutputField(desc="One-word answer")

cot = dspy.ChainOfThought(Counter)      # << program fed to Evaluate

# ---------- 3. Custom metric using an LLM judge ----------
class Assess(dspy.Signature):
    assessed_text       = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer   = dspy.OutputField()

def metric(gold, pred, trace=None):
    q, gold_ans, pred_ans = gold.question, gold.answer, pred.answer
    prompt = f"The text should answer `{q}` with `{gold_ans}`. Does the assessed text contain this answer?"
    judge  = dspy.Predict(Assess)(assessed_text=pred_ans, assessment_question=prompt)
    is_ok  = judge.assessment_answer
    # If DSPy is compiling (trace!=None) return bool, else return score in [0,1]
    return bool(is_ok) if trace is not None else (1.0 if is_ok else 0.0)

# ---------- 4. Run evaluation ----------
with mlflow.start_run(run_name="CoT Evaluation"):
    evaluator = dspy.evaluate.Evaluate(
        devset=train_set,
        return_all_scores=True,
        return_outputs=True,
        show_progress=True
    )
    agg_score, outputs, scores = evaluator(cot, metric=metric)

    mlflow.log_metric("custom_metric", agg_score)
    mlflow.log_table(
        {
            "question": [ex.question for ex in train_set],
            "answer":   [ex.answer   for ex in train_set],
            "output":   [o[0].answer    for o in outputs],
            "metric":   scores,
        },
        artifact_file="eval_results.json",
    )


# COMMAND ----------

# MAGIC %md ### Optimize DSPy module using Compilation

# COMMAND ----------

# Enable auto-tracing for compilation
mlflow.dspy.autolog(log_traces_from_compile=True)

# Optimize the DSPy program as usual
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)
optimized = tp.compile(cot, trainset=train_set)

# COMMAND ----------

# MAGIC %md ### Run optimized module

# COMMAND ----------

result = optimized(question="Who is father of Hercules?")
print(result)

# COMMAND ----------

optimized.inspect_history()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Define a prompt optimization evaluation function
# MAGIC
# MAGIC The following defines and implements a function to evaluate if the responses from the program are correct.
# MAGIC Mosaic Agent Evaluation ([AWS](https://docs.databricks.com/generative-ai/tutorials/agent-framework-notebook.html) | [Azure](https://learn.microsoft.com/azure/databricks/generative-ai/tutorials/agent-framework-notebook)) is an ideal tool for this purpose.

# COMMAND ----------

# import mlflow
# import pandas as pd

# def evalute_using_mosaic_agent(example, pred, trace=None):
#     # Running evaluation using the Mosaic Agent Evaluation
#     result = mlflow.evaluate(
#         data=pd.DataFrame([example.toDict() | {"response": pred["response"]}]),
#         model_type="databricks-agent",
#     )
#     return (
#         # Using the Mosaic Agent Evaluation "correctness rating" metric to determine if the response is correct
#         result.tables["eval_results"]["response/llm_judged/correctness/rating"][0]
#         == "yes"
#     )

# COMMAND ----------

# MAGIC %md
# MAGIC ####Run optimization
# MAGIC
# MAGIC Now, the final step is to run the optimization. DSPy offers several [optimizers](https://dspy-docs.vercel.app/docs/building-blocks/optimizers), this example uses the `BootstrapFewShot` optimizer. The `BootstrapFewShot` optimizer selects the best `few-shot examples` for all the stages of the DSPy program, but in this notebook you only use one stage.  The examples are obtained from the training set labels (`expected_response`) and the evaluation executions.  For more information about this and other optimizers, see the [DSPy documentaion](https://dspy-docs.vercel.app/docs/building-blocks/optimizers).

# COMMAND ----------

# from dspy.evaluate.evaluate import Evaluate
# from dspy.teleprompt import BootstrapFewShot

# # Set up a bootstrap optimizer, which optimizes the RAG program.
# optimizer = BootstrapFewShot(
#     metric=evalute_using_mosaic_agent, # Use defined evaluation function
#     max_bootstrapped_demos=4, # Max number of examples obtained from running the train set
#     max_labeled_demos=8 # Max number of examples obtained from labels in the train set
# )

# # Start a new MLflow run to track all evaluation metrics
# with mlflow.start_run(run_name="dspy_rag_optimization"):
#     # Optimize the program by identifying the best few-shot examples for the prompt used by the `response_generator` step
#     optimized_rag = optimizer.compile(rag, trainset=train_set)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Run the optimized DSPy module
# MAGIC
# MAGIC Try the tricky question again:

# COMMAND ----------

# result = optimized_rag("Who is father of Hercules?")
# print(result.response)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Inspect the prompt used by the optimized program
# MAGIC
# MAGIC When inspecting the prompt generated from the optimized program, the few-shot examples are added by DSPy:

# COMMAND ----------

# optimized_rag.lm.inspect_history()