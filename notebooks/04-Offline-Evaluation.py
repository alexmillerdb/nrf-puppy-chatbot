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

eval_dataset = spark.table(f"{catalog}.{db}.web_style_data_embedded")

display(eval_dataset)

# COMMAND ----------

df_qa = (spark.read.table('evaluation_dataset')
                  .selectExpr('question as inputs', 'answer as targets')
                  .where("targets is not null")
                  .sample(fraction=0.005, seed=40)) #small sample for interactive demo

df_qa_with_preds = df_qa.withColumn('preds', predict_answer(col('inputs'))).cache()

display(df_qa_with_preds)

# COMMAND ----------

# MAGIC %md ## Tag model as production ready if it meets expectations

# COMMAND ----------

client = MlflowClient()
client.set_registered_model_alias(name=model_name, alias="prod", version=model_version_to_evaluate)
