# Databricks notebook source
# MAGIC %pip install databricks-genai-inference databricks-sdk==0.12.0 mlflow==2.9.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## ADD TO CONFIG

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME = "petm_genai_chatbot"
catalog = "main"
db = "databricks_petm_chatbot"
source_table = "web_style_data_embedded"

# COMMAND ----------

# MAGIC %md ## ADD TO HELPER FUNCTIONS

# COMMAND ----------

import os
import requests
import time

class EndpointApiClient:
    def __init__(self):
        self.base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def create_inference_endpoint(self, endpoint_name, served_models, auto_capture_config = None):
        data = {"name": endpoint_name, "config": {"served_models": served_models, "auto_capture_config": auto_capture_config}}
        return self._post("api/2.0/serving-endpoints", data)

    def get_inference_endpoint(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}", allow_error=True)
      
      
    def inference_endpoint_exists(self, endpoint_name):
      ep = self.get_inference_endpoint(endpoint_name)
      if 'error_code' in ep and ep['error_code'] == 'RESOURCE_DOES_NOT_EXIST':
          return False
      if 'error_code' in ep and ep['error_code'] != 'RESOURCE_DOES_NOT_EXIST':
          raise Exception(f"enpoint exists ? {ep}")
      return True

    def create_endpoint_if_not_exists(self, endpoint_name, model_name, model_version, workload_size, scale_to_zero_enabled=True, wait_start=True, auto_capture_config = None, environment_vars = {}):
      models = [{
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero_enabled,
            "environment_vars": environment_vars
      }]
      if not self.inference_endpoint_exists(endpoint_name):
        r = self.create_inference_endpoint(endpoint_name, models, auto_capture_config)
      #Make sure we have the proper version deployed
      else:
        ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            self.wait_endpoint_start(endpoint_name)
            ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            model_deployed = ep['pending_config']['served_models'][0]
            print(f"Error with the model deployed: {model_deployed} - state {ep['state']}")
        else:
            model_deployed = ep['config']['served_models'][0]
        if model_deployed['model_version'] != model_version:
          print(f"Current model is version {model_deployed['model_version']}. Updating to {model_version}...")
          u = self.update_model_endpoint(endpoint_name, {"served_models": models})
      if wait_start:
        self.wait_endpoint_start(endpoint_name)
      
      
    def list_inference_endpoints(self):
        return self._get("api/2.0/serving-endpoints")

    def update_model_endpoint(self, endpoint_name, conf):
        return self._put(f"api/2.0/serving-endpoints/{endpoint_name}/config", conf)

    def delete_inference_endpoint(self, endpoint_name):
        return self._delete(f"api/2.0/serving-endpoints/{endpoint_name}")

    def wait_endpoint_start(self, endpoint_name):
      i = 0
      while self.get_inference_endpoint(endpoint_name)['state']['config_update'] == "IN_PROGRESS" and i < 500:
        if i % 10 == 0:
          print("waiting for endpoint to build model image and start...")
        time.sleep(10)
        i += 1
      ep = self.get_inference_endpoint(endpoint_name)
      if ep['state'].get("ready", None) != "READY":
        print(f"Error creating the endpoint: {ep}")
        
      
    # Making predictions

    def query_inference_endpoint(self, endpoint_name, data):
        return self._post(f"realtime-inference/{endpoint_name}/invocations", data)

    # Debugging

    def get_served_model_build_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/build-logs"
        )

    def get_served_model_server_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/logs"
        )

    def get_inference_endpoint_events(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}/events")

    def _get(self, uri, data = {}, allow_error = False):
        r = requests.get(f"{self.base_url}/{uri}", params=data, headers=self.headers)
        return self._process(r, allow_error)

    def _post(self, uri, data = {}, allow_error = False):
        return self._process(requests.post(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _put(self, uri, data = {}, allow_error = False):
        return self._process(requests.put(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _delete(self, uri, data = {}, allow_error = False):
        return self._process(requests.delete(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _process(self, r, allow_error = False):
      if r.status_code == 500 or r.status_code == 403 or not allow_error:
        print(r.text)
        r.raise_for_status()
      return r.json()

# COMMAND ----------

import urllib
import json
import mlflow
from mlflow import MlflowClient

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

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

# MAGIC %md ### Perform load testing

# COMMAND ----------

# MAGIC %md Create generated dataset

# COMMAND ----------

from databricks_genai_inference import ChatCompletion

response = ChatCompletion.create(model="llama-2-70b-chat",
                                 messages=[{"role": "system", "content": "You are an AI assistant that specializes in PetSmart and Dogs. Your task is to generate 20 questions related to dogs food preferences, food recommendations, toys, treats. The questions should also include specific characteristics about the dog such as lifestage (puppy, adult, senior), dietary restrictions, breeds (German Shepherds), activity levels, and more."},
                                 {"role": "user", "content": "Generate questions based on life stages (puppy, adult, senior), dietary restrictions, popular dog breeds, active dogs, and potty training."}],
                                 max_tokens=1500)
print(f"response.message:{response.message}")

# COMMAND ----------

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

import pandas as pd
questions_df = spark.createDataFrame(pd.DataFrame({"question": question_list}))
display(questions_df)

# COMMAND ----------

# MAGIC %md Run load testing

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

send_requests_to_endpoint_and_wait_for_payload_to_be_available(endpoint_name=serving_endpoint_name, question_df=questions_df, limit=40)

# COMMAND ----------

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import re

# Define the asynchronous function to make API calls
async def llama(session, url, token, text, semaphore):
    async with semaphore:  # Acquire a spot in the semaphore
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        prompt = get_prompt(text)  # Ensure this function is defined
        data_input = get_data_input(prompt=prompt, max_tokens=1500, temperature=0.8)  # Ensure this function is defined
        
        async with session.post(url=url, json=data_input, headers=headers) as response:
            return await response.json()

async def main(url, token, vegetarian_topics, schema_keys, max_concurrent_tasks):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)  # Control concurrency
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(2500):  # Adjust the range as needed
            topic = np.random.choice(vegetarian_topics)
            task = asyncio.create_task(llama(session, url, token, topic, semaphore))
            tasks.append(task)
        
        raw_responses = await asyncio.gather(*tasks)
        results = []
        for resp in raw_responses:
            try:
                data = process_result(resp["predictions"][0]["candidates"][0]["text"])
                if all(key in data and isinstance(data[key], str) for key in schema_keys):
                    results.append(data)
                else:
                    print("Dictionary does not match schema.")
            except Exception as e:
                continue

        return results

# Define your URL, token, and other variables
url = 'https://adb-984752964297111.11.azuredatabricks.net/serving-endpoints/trl_llms_for_good_data/invocations'
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

schema_keys = ["question", "good_answer", "bad_answer"]

# vegetarian_topics = ["Nutritional Benefits of Vegetarianism", "Plant-Based Protein Sources", "Vegetarian Meal Planning", "Vegetarian Cooking Techniques",
#                      "Transitioning to Vegetarianism", "Global Vegetarian Dishes", "Seasonal Vegetarian Recipes", "Vegetarian Kids' Meals", "Vegetarian and Vegan Substitutes",
#                      "Environmental Impact of Vegetarianism","Vegetarian Diet Myths and Facts","Special Diets and Vegetarianism","Dining Out as a Vegetarian",
#                      "Vegetarian Athlete Nutrition","Homemade Vegetarian Snacks","Budget-Friendly Vegetarian Meals","Wine Pairing with Vegetarian Food",
#                      "Vegetarianism in Different Cultures","Vegetarian Protein for Bodybuilding","Vegetarian Holiday Recipes"]

vegetarian_topics = ["Protein Sources", "Meal Planning", "Cooking Techniques", "Global Cuisine Dishses", "Seasonal Meat Recipes", "Athlete Nutrition",
                     "Wine Pairing with Food", "Holiday Recipes", "Nutritional Recipes that include Meat", "Unhealthy recipes", "Downsides of Vegetarianism"]

# ... Define URL, token, and call main ...

max_concurrent_tasks = 15  # Set this to control concurrency
results = await main(url, DATABRICKS_TOKEN, vegetarian_topics, schema_keys, max_concurrent_tasks)

# ... Convert results to DataFrame ...
# Create a DataFrame from the results
df2 = pd.DataFrame(results)
df2.head()
