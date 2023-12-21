# Databricks notebook source
# MAGIC %pip install databricks-sdk==0.12.0 databricks-genai-inference==0.1.1 mlflow==2.9.0 textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 langchain==0.0.344 databricks-vectorsearch==0.22 transformers==4.30.2 torch==2.0.1 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Using Llama2-70b as "judge"

# COMMAND ----------

from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")

# try:
#     endpoint_name  = "dbdemos-azure-openai"
#     deploy_client.create_endpoint(
#         name=endpoint_name,
#         config={
#             "served_entities": [
#                 {
#                     "name": endpoint_name,
#                     "external_model": {
#                         "name": "gpt-35-turbo",
#                         "provider": "openai",
#                         "task": "llm/v1/chat",
#                         "openai_config": {
#                             "openai_api_type": "azure",
#                             "openai_api_key": "{{secrets/dbdemos/azure-openai}}", #Replace with your own azure open ai key
#                             "openai_deployment_name": "dbdemo-gpt35",
#                             "openai_api_base": "https://dbdemos-open-ai.openai.azure.com/",
#                             "openai_api_version": "2023-05-15"
#                         }
#                     }
#                 }
#             ]
#         }
#     )
# except Exception as e:
#     if 'RESOURCE_ALREADY_EXISTS' in str(e):
#         print('Endpoint already exists')
#     else:
#         print(f"Couldn't create the external endpoint with Azure OpenAI: {e}. Will fallback to llama2-70-B as judge. Consider using a stronger model as a judge.")
#         endpoint_name = "databricks-llama-2-70b-chat"

endpoint_name = "databricks-llama-2-70b-chat"

#Let's query our external model endpoint
answer_test = deploy_client.predict(endpoint=endpoint_name, inputs={"messages": [{"role": "user", "content": "What is PetSmart?"}]})
answer_test['choices'][0]['message']['content']

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Offline LLM evaluation
# MAGIC
# MAGIC We will start with offline evaluation, scoring our model before its deployment. This requires a set of questions we want to ask to our model.
# MAGIC
# MAGIC In our case, we are fortunate enough to have a labeled training set (questions+answers)  with state-of-the-art technical answers from our Databricks support team. Let's leverage it so we can compare our RAG predictions and ground-truth answers in MLflow.
# MAGIC
# MAGIC **Note**: This is optional! We can benefit from the LLMs-as-a-Judge approach without ground-truth labels. This is typically the case if you want to evaluate "live" models answering any customer questions

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

from pyspark.sql import functions as F
import os 

catalog = "main"
db = "databricks_petm_chatbot"

os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
model_name = f"{catalog}.{db}.petm_chatbot_model"
model_version_to_evaluate = get_latest_model_version(model_name)
print(f"Model version to evaluate: {model_version_to_evaluate}")
mlflow.set_registry_uri("databricks-uc")
rag_model = mlflow.langchain.load_model(f"models:/{model_name}/{model_version_to_evaluate}")

@F.pandas_udf("string")
def predict_answer(questions):
    def answer_question(question):
        dialog = {"messages": [{"role": "user", "content": question}]}
        return rag_model.invoke(dialog)['result']
    return questions.apply(answer_question)

# COMMAND ----------

# MAGIC %md Create eval dataset:
# MAGIC - Work with PetSmart team on creating eval dataset based on human labeler or LLM generated

# COMMAND ----------

df_qa = spark.table(f"{catalog}.{db}.faqs_chunked") \
    .filter(F.col("num_chunks") == 1) \
    .withColumn("questions_plus_context", F.concat(F.lit("PetSmart: "), F.col("context"), F.lit(" "), F.col("question"))) \
    .selectExpr('questions_plus_context as inputs', 'answer as targets') \
    .limit(20)

df_qa_with_preds = df_qa.withColumn('preds', predict_answer(F.col('inputs'))).cache()

print(df_qa_with_preds.count())
display(df_qa_with_preds)

# COMMAND ----------

# df_qa = (spark.read.table('evaluation_dataset')
#                   .selectExpr('question as inputs', 'answer as targets')
#                   .where("targets is not null")
#                   .sample(fraction=0.005, seed=40)) #small sample for interactive demo

# df_qa_with_preds = df_qa.withColumn('preds', predict_answer(col('inputs'))).cache()

# display(df_qa_with_preds)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##LLMs-as-a-judge: automated LLM evaluation with out of the box and custom GenAI metrics
# MAGIC
# MAGIC MLflow 2.8 provides out of the box GenAI metrics and enables us to make our own GenAI metrics:
# MAGIC - Mlflow will automatically compute relevant task-related metrics. In our case, `model_type='question-answering'` will add the `toxicity` and `token_count` metrics.
# MAGIC - Then, we can import out of the box metrics provided by MLflow 2.8. Let's benefit from our ground-truth labels by computing the `answer_correctness` metric. 
# MAGIC - Finally, we can define customer metrics. Here, creativity is the only limit. In our demo, we will evaluate the `professionalism` of our Q&A chatbot.

# COMMAND ----------

from mlflow.metrics.genai.metric_definitions import answer_correctness
from mlflow.metrics.genai import make_genai_metric, EvaluationExample

# Because we have our labels (answers) within the evaluation dataset, we can evaluate the answer correctness as part of our metric. Again, this is optional.
answer_correctness_metrics = answer_correctness(model=f"endpoints:/{endpoint_name}")
print(answer_correctness_metrics)

# COMMAND ----------

# MAGIC %md Adding custom professionalism metric

# COMMAND ----------

professionalism_example = EvaluationExample(
    input="What is MLflow?",
    output=(
        "MLflow is like your friendly neighborhood toolkit for managing your machine learning projects. It helps "
        "you track experiments, package your code and models, and collaborate with your team, making the whole ML "
        "workflow smoother. It's like your Swiss Army knife for machine learning!"
    ),
    score=2,
    justification=(
        "The response is written in a casual tone. It uses contractions, filler words such as 'like', and "
        "exclamation points, which make it sound less professional. "
    )
)

professionalism = make_genai_metric(
    name="professionalism",
    definition=(
        "Professionalism refers to the use of a formal, respectful, and appropriate style of communication that is "
        "tailored to the context and audience. It often involves avoiding overly casual language, slang, or "
        "colloquialisms, and instead using clear, concise, and respectful language."
    ),
    grading_prompt=(
        "Professionalism: If the answer is written using a professional tone, below are the details for different scores: "
        "- Score 1: Language is extremely casual, informal, and may include slang or colloquialisms. Not suitable for "
        "professional contexts."
        "- Score 2: Language is casual but generally respectful and avoids strong informality or slang. Acceptable in "
        "some informal professional settings."
        "- Score 3: Language is overall formal but still have casual words/phrases. Borderline for professional contexts."
        "- Score 4: Language is balanced and avoids extreme informality or formality. Suitable for most professional contexts. "
        "- Score 5: Language is noticeably formal, respectful, and avoids casual elements. Appropriate for formal "
        "business or academic settings. "
    ),
    model=f"endpoints:/{endpoint_name}",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance"],
    examples=[professionalism_example],
    greater_is_better=True
)

print(professionalism)

# COMMAND ----------

# MAGIC %md Start evaluation run

# COMMAND ----------

from mlflow.deployments import set_deployments_target

set_deployments_target("databricks")

#This will automatically log all
with mlflow.start_run(run_name="petm_chatbot_rag") as run:
    eval_results = mlflow.evaluate(data = df_qa_with_preds.toPandas(), # evaluation data,
                                   model_type="question-answering", # toxicity and token_count will be evaluated   
                                   predictions="preds", # prediction column_name from eval_df
                                   targets = "targets",
                                   extra_metrics=[answer_correctness_metrics, professionalism])
    
eval_results.metrics

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Visualizations of GenAI metrics produced by Llama2-70B

# COMMAND ----------

df_genai_metrics = eval_results.tables["eval_results_table"]
display(df_genai_metrics)

# COMMAND ----------

import plotly.express as px
px.histogram(df_genai_metrics, x="token_count", labels={"token_count": "Token Count"}, title="Distribution of Token Counts in Model Responses")

# COMMAND ----------

# Counting the occurrences of each answer correctness score
px.bar(df_genai_metrics['answer_correctness/v1/score'].value_counts(), title='Answer Correctness Score Distribution')

# COMMAND ----------

df_genai_metrics['toxicity'] = df_genai_metrics['toxicity/v1/score'] * 100
fig = px.scatter(df_genai_metrics, x='toxicity', y='answer_correctness/v1/score', title='Toxicity vs Correctness', size=[10]*len(df_genai_metrics))
fig.update_xaxes(tickformat=".2f")

# COMMAND ----------

df_genai_metrics[df_genai_metrics['answer_correctness/v1/score'] == 3]

# COMMAND ----------

# MAGIC %md ## Tag model as production ready if it meets expectations

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(name=model_name, alias="prod", version=model_version_to_evaluate)
