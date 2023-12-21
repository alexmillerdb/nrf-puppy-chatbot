# Databricks notebook source
# MAGIC %pip install databricks-genai-inference databricks-sdk==0.12.0 mlflow==2.9.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../configs/00-config

# COMMAND ----------

# MAGIC %md ### Deploy Model Serving Endpoint

# COMMAND ----------

import urllib
import json
import mlflow
from mlflow import MlflowClient

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
from functions.model_serving import EndpointApiClient

mlflow.set_registry_uri('databricks-uc')
client = MlflowClient()
model_name = f"{catalog}.{db}.petm_chatbot_model"
serving_endpoint_name = f"petm_chatbot_endpoint_{catalog}_{db}"[:63]
latest_model = client.get_model_version_by_alias(model_name, "prod")

w = WorkspaceClient()
#TODO: use the sdk once model serving is available.
serving_client = EndpointApiClient()
# Start the endpoint using the REST API (you can do it using the UI directly)
auto_capture_config = {
    "catalog_name": catalog,
    "schema_name": db,
    "table_name_prefix": serving_endpoint_name
    }
environment_vars={"DATABRICKS_TOKEN": "{{secrets/nrf-petm-chatbot/rag_sp_token}}"}
serving_client.create_endpoint_if_not_exists(serving_endpoint_name, 
                                             model_name=model_name, 
                                             model_version = latest_model.version, 
                                             workload_size="Small", 
                                             scale_to_zero_enabled=True, 
                                             wait_start = True, 
                                             auto_capture_config=auto_capture_config, environment_vars=environment_vars)

# COMMAND ----------

displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md Test endpoint with query

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import DataframeSplitInput

df_split = DataframeSplitInput(columns=["messages"],
                               data=[[ {"messages": [{"role": "user", "content": "What is PetSmart?"}, 
                                                     {"role": "assistant", "content": "PetSmart is a Pet Specialty retailer."}, 
                                                     {"role": "user", "content": "Does PetSmart sell dog food for small breed dogs like chihuahuas?"}
                                                    ]}]])
w = WorkspaceClient()
w.serving_endpoints.query(serving_endpoint_name, dataframe_split=df_split)

# COMMAND ----------

df_split = DataframeSplitInput(columns=["messages"],
                               data=[[ {"messages": [{"role": "user", "content": "What is PetSmart?"}, 
                                                     {"role": "assistant", "content": "PetSmart is a Pet Specialty retailer."}, 
                                                     {"role": "user", "content": "I have German Shepherd adult dog that is sensitive to Chicken treats. Are there other treats that do not have chicken you could recommend?"}
                                                    ]}]])
w = WorkspaceClient()
w.serving_endpoints.query(serving_endpoint_name, dataframe_split=df_split)

# COMMAND ----------

df_split = DataframeSplitInput(columns=["messages"],
                               data=[[ {"messages": [{"role": "user", "content": "Does PetSmart have a loyalty program?"}, 
                                                     {"role": "assistant", "content": "Yes, PetSmart has a loyalty program called Treats."}, 
                                                     {"role": "user", "content": "What are the main benefits of the Treats program?"}
                                                    ]}]])
w = WorkspaceClient()
response = w.serving_endpoints.query(serving_endpoint_name, dataframe_split=df_split)

# COMMAND ----------

dialog = [{"role": "user", "content": "Does PetSmart have a loyalty program?"}, 
                                                     {"role": "assistant", "content": "Yes, PetSmart has a loyalty program called Treats."}, 
                                                     {"role": "user", "content": "What are the main benefits of the Treats program?"}
                                                    ]

# COMMAND ----------

def display_chat(chat_history, response):
  def user_message_html(message):
    return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #c2efff; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px;">
        {message}
      </div>"""
  def assistant_message_html(message):
    return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #e3f6fc; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
        <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/>
        {message}
      </div>"""
  chat_history_html = "".join([user_message_html(m["content"]) if m["role"] == "user" else assistant_message_html(m["content"]) for m in chat_history])
  answer = response["result"].replace('\n', '<br/>')
  sources_html = ("<br/><br/><br/><strong>Sources:</strong><br/> <ul>" + '\n'.join([f"""<li><a href="{s}">{s}</a></li>""" for s in response["sources"]]) + "</ul>") if response["sources"] else ""
  response_html = f"""{answer}{sources_html}"""

  displayHTML(chat_history_html + assistant_message_html(response_html))

# COMMAND ----------

display_chat(dialog, response.predictions[0])

# COMMAND ----------

# MAGIC %md ### Perform load testing

# COMMAND ----------

# MAGIC %md Create generated dataset

# COMMAND ----------

from databricks_genai_inference import ChatCompletion
import re

question_list = []

for i in range(2):
    # generate questions from Llama2
    response = ChatCompletion.create(model="llama-2-70b-chat",
                                    messages=[{"role": "system", "content": "You are an AI assistant that specializes in PetSmart and Dogs. Your task is to generate 20 questions related to dogs food preferences, food recommendations, toys, treats. The questions should also include specific characteristics about the dog such as lifestage (puppy, adult, senior), dietary restrictions, breeds (German Shepherds), activity levels, and more."},
                                    {"role": "user", "content": "Generate questions based on life stages (puppy, adult, senior), dietary restrictions, popular dog breeds, active dogs, and potty training."}],
                                    max_tokens=1500)
    bulleted_list = [line.strip() for line in response.message.split('\n') if line.strip().startswith(tuple('123456789'))]
    cleaned_list = [re.sub(r'^[0-9.]+\s*', '', item).strip() for item in bulleted_list]
    question_list += cleaned_list
    # question_list.append(cleaned_list)

print(len(question_list))
question_list

# COMMAND ----------

for i in range(3):
    # generate questions from Llama2
    response = ChatCompletion.create(model="llama-2-70b-chat",
                                    messages=[{"role": "system", "content": "You are an AI assistant that specializes in PetSmart and Dogs. Your task is to generate 20 questions related to puppy and dog care such as potty training, behavior training, and taking care of your pet (grooming, bathing, feeding schedules). The questions should also include specific questions about how to take care of puppies and dogs."},
                                    {"role": "user", "content": "Generate questions based on retail puppy care, grooming, bathing, potty training."}],
                                    max_tokens=1500)
    bulleted_list = [line.strip() for line in response.message.split('\n') if line.strip().startswith(tuple('123456789'))]
    cleaned_list = [re.sub(r'^[0-9.]+\s*', '', item).strip() for item in bulleted_list]
    question_list += cleaned_list
    # question_list.append(cleaned_list)

print(len(question_list))
question_list

# COMMAND ----------

import pandas as pd
question_df = spark.createDataFrame(pd.DataFrame({"question": question_list}))
display(question_df)

# COMMAND ----------

# MAGIC %md ### Run load testing

# COMMAND ----------

# MAGIC %md #### Load testing using ThreadPoolExecutor

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor

def send_requests_to_endpoint_and_wait_for_payload_to_be_available(endpoint_name, question_df, limit=50):
  print(f'Sending {limit} requests to the endpoint {endpoint_name}, this will takes a few seconds...')
  #send some requests
  from databricks.sdk import WorkspaceClient
  w = WorkspaceClient()
  def answer_question(question):
    df_split = DataframeSplitInput(columns=["messages"],
                                   data=[[ {"messages": [{"role": "user", "content": question}]} ]])
    answer = w.serving_endpoints.query(endpoint_name, dataframe_split=df_split)
    return answer.predictions[0]

  df_questions = question_df.limit(limit).toPandas()['question']
  with ThreadPoolExecutor(max_workers=5) as executor:
      results = list(executor.map(answer_question, df_questions))
  print(results)

  #Wait for the inference table to be populated
  print('Waiting for the inference to be in the Inference table, this can take a few seconds...')
  from time import sleep
  for i in range(10):
    if not spark.table(f'{endpoint_name}_payload').count() < len(df_questions):
      break
    sleep(10)

# COMMAND ----------

from functions.model_serving import send_requests_to_endpoint_and_wait_for_payload_to_be_available

send_requests_to_endpoint_and_wait_for_payload_to_be_available(endpoint_name=serving_endpoint_name, question_df=question_df, limit=20)

# COMMAND ----------

# MAGIC %md #### Load testing using `asyncio` and `aiohttp`

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

os.environ["DATABRICKS_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def get_response_data(question):
    return {
        'columns': ['messages'],
        'data': [
            [
                {
                    'messages': [
                        {
                            'role': 'user',
                            'content': f"{question}"
                        }
                    ]
                }
            ]
        ]
    }

def score_model(dataset):
    url = 'https://adb-6042569476650449.9.azuredatabricks.net/serving-endpoints/petm_chatbot_endpoint_main_databricks_petm_chatbot/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()

# COMMAND ----------

df_questions = pd.DataFrame({"question": question_list})['question']
df_questions[0]

# COMMAND ----------

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import re

url = 'https://adb-6042569476650449.9.azuredatabricks.net/serving-endpoints/petm_chatbot_endpoint_main_databricks_petm_chatbot/invocations'

# Define the asynchronous function to make API calls
async def llama(session, url, question, semaphore):
    async with semaphore:  # Acquire a spot in the semaphore
        url = 'https://adb-6042569476650449.9.azuredatabricks.net/serving-endpoints/petm_chatbot_endpoint_main_databricks_petm_chatbot/invocations'
        headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
        dataset = get_response_data(question=question)
        data_json = {'dataframe_split': dataset}
        
        async with session.post(url=url, json=data_json, headers=headers) as response:
            return await response.json()

async def main(url, questions, max_concurrent_tasks):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)  # Control concurrency
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(100):  # Adjust the range as needed
            question = np.random.choice(questions)
            task = asyncio.create_task(llama(session, url, question, semaphore))
            tasks.append(task)
        
        raw_responses = await asyncio.gather(*tasks)
        results = []
        for resp in raw_responses:
            try:
                results.append(resp)
            except Exception as e:
                continue

        return results

max_concurrent_tasks = 10  # Set this to control concurrency
results = await main(url, question_list, max_concurrent_tasks)

df2 = pd.DataFrame(results)
df2.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load test results:
# MAGIC Max concurrency of 15: 
# MAGIC - Latency (P50): 45-60K MS
# MAGIC - RPS 1.7 to 3.3
# MAGIC - Request Error Rates PS: 1.7 to 3.5
# MAGIC - Provisioned Throughput 4
# MAGIC
# MAGIC Max concurrency of 5:
# MAGIC - Latency (P50): 16-18K MS
# MAGIC - RPS 0.2 to 0.3
# MAGIC - Request Error Rates PS: 0
# MAGIC - Provisioned Throughput 4
# MAGIC
# MAGIC Max concurrency of 10:
# MAGIC - Latency (P50): 34-39K MS
# MAGIC - RPS 0.15 to 0.27
# MAGIC - Request Error Rates PS: 0
# MAGIC - Provisioned Throughput 4
