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
source_table = "petm_data_embedded"
secret_scope_name = "NRF_PUPPY_CHATBOT"
secret_key_name = "PAT"

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
# serving_endpoint_name = f"nrf_puppy_chatbot_test"
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
environment_vars = { "DATABRICKS_TOKEN": f"{{{{secrets/{secret_scope_name}/{secret_key_name}}}}}" }
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
question_df.write.format("delta").mode("overwrite").saveAsTable("main.databricks_petm_chatbot.generated_qa_questions")

# COMMAND ----------

# MAGIC %md ## Run load testing

# COMMAND ----------

question_df = spark.table("main.databricks_petm_chatbot.generated_qa_questions")
display(question_df)

# COMMAND ----------

# MAGIC %md ### Load testing using ThreadPoolExecutor

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

send_requests_to_endpoint_and_wait_for_payload_to_be_available(endpoint_name=serving_endpoint_name, question_df=question_df, limit=81)

# COMMAND ----------

# MAGIC %md ### Load testing using `asyncio` and `aiohttp`

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

os.environ["DATABRICKS_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)

os.environ["DATABRICKS_HOST"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
)

os.environ["ENDPOINT_URL"] = os.path.join(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get(),
    "serving-endpoints",
    serving_endpoint_name,
    "invocations",
)


def get_response_data(question: str):
    
    return {
        "columns": ["messages"],
        "data": [[{"messages": [{"role": "user", "content": f"{question}"}]}]],
    }


def score_model(dataset: dict):
    
    url = os.environ.get("ENDPOINT_URL")
    headers = {
        "Authorization": f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
        "Content-Type": "application/json",
    }
    ds_dict = {"dataframe_split": dataset}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status {response.status_code}, {response.text}"
        )

    return response.json()

# COMMAND ----------

request = get_response_data(question="How often should I bathe my puppy, and what shampoos and conditioners are safe to use?")
response = score_model(dataset=request)
response

# COMMAND ----------

# MAGIC %md ### Mimic a chat history

# COMMAND ----------

df_questions = question_df.toPandas()
question_list = df_questions["question"].to_list()
question_list

# COMMAND ----------

chat_session = []

for question in df_questions.iloc[0:10]['question']:
    request = get_response_data(question=question)
    response = score_model(dataset=request)
    chat_session += [{"role": "user", "content": f"{question}"},
                     {"role": "assistant", "content": f"{response['predictions'][0]['result']}"}]
    
    print(chat_session)

# COMMAND ----------

len(chat_session)

# COMMAND ----------

dialog_test = {
    "columns": ["messages"],
    "data": [
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What are the best food options for a puppy German Shepherd with sensitive skin?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Based on the information provided, the best food options for a puppy German Shepherd with sensitive skin are Royal Canin Breed Health Nutrition German Shepherd Puppy Dry Dog Food and Hill's Prescription Diet Derm Complete Puppy Environmental/Food Sensitivities Dry Dog Food. Both foods are specifically formulated to support the skin's barriers against environmental irritants and provide nutritional support for building skin health. Additionally, they both contain ingredients that are easy to digest and are designed to promote healthy digestion. It's important to consult with a veterinarian to determine the best food option for your puppy's specific needs.",
                    },
                    {
                        "role": "user",
                        "content": "How can I transition my adult dog from a puppy food to an adult food?",
                    },
                    {
                        "role": "assistant",
                        "content": "  According to the information provided, it is recommended to gradually transition your adult dog from puppy food to adult food over a 7-10 day period. Replace 25% of the current diet with the new diet every 2-3 days until they are fully transitioned. It's also recommended to consult with your veterinarian for specific recommendations on transitioning your dog's diet and to ensure that the dog is healthy enough for the transition.",
                    },
                    {
                        "role": "user",
                        "content": "What are the best senior dog food options for a dog with joint issues?",
                    },
                    {
                        "role": "assistant",
                        "content": "Eukanuba Senior Small Breed Dry Dog Food and Eukanuba Senior Medium Breed Dry Dog Food are both good options for senior dogs with joint issues. They both contain glucosamine and chondroitin sulfate, which support joint health and mobility. Additionally, they have optimal levels of fats and carbohydrates to help maintain energy levels and support active lifestyles. They also provide immune system support and dental care.",
                    },
                    {
                        "role": "user",
                        "content": "My dog has a sensitive stomach, what are some limited ingredient dog food options?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Limited ingredient dog food options for sensitive stomachs include:\n\n1. Hill's Science Diet Sensitive Stomach & Skin Small Breed Adult Dry Dog Food - Chicken & Barley\n2. Purina Pro Plan Specialized Adult Dry Dog Food - Sensitive Skin & Stomach, Turkey & Oatmeal\n3. Purina Pro Plan Sensitive Skin & Stomach Adult Dry Dog Food - Salmon & Rice",
                    },
                    {
                        "role": "user",
                        "content": "What are the best dog treats for a puppy in training?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Wellness Soft Puppy Bites Natural Dog Treats - Natural, Lamb & Salmon.",
                    },
                    {
                        "role": "user",
                        "content": "Can you recommend a high-protein dog food for my active German Shepherd?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Sure! Based on your dog's activity level and breed size, I would recommend the Purina Pro Plan Sport All Life Stage Dry Dog Food - High Energy, High Protein, Salmon & Rice. It has a high protein content of 30% and 20% fat, which will support your dog's metabolic needs and help maintain lean, strong muscles. Additionally, it contains EPA and glucosamine for joint health and mobility, as well as guaranteed live probiotics for digestive and immune health. This recipe is designed for highly active dogs, including hunting and sporting dogs, and competitive canine athletes, making it a great fit for your German Shepherd.",
                    },
                    {
                        "role": "user",
                        "content": "What are the best toys for a puppy going through teething?",
                    },
                    {
                        "role": "assistant",
                        "content": "The best toys for a puppy going through teething are the Playology Puppy Teething Ball Dog Toy - Beef, the Playology Puppy Teething Ball Dog Toy - Peanut Butter, and the Playology Puppy Teething Ring Dog Toy - Peanut Butter. These toys are designed specifically for teething puppies and are made with natural rubber that is gentle on their gums. They also have unique textures and sounds that make them enticing for puppies to chew on, and they are built to last with puncture-resistant construction. Additionally, they use Playology's scent technology to drive 7 times more engagement than unscented toys, providing hours of stimulating playtime for teething puppies.",
                    },
                    {
                        "role": "user",
                        "content": "How often should I feed my senior dog to prevent weight gain?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Once or twice a day. You can start with the recommended amount on the package and adjust accordingly to maintain your dog's weight. It's also important to consult with your veterinarian for specific feeding instructions tailored to your dog's individual needs.",
                    },
                    {
                        "role": "user",
                        "content": "What are some hypoallergenic dog food options for a dog with food allergies?",
                    },
                    {
                        "role": "assistant",
                        "content": "  There are several hypoallergenic dog food options available for dogs with food allergies. Some examples include:\n\n1. Limited Ingredient Diets (LIDs): These dog foods contain a minimal number of ingredients, which can help reduce the risk of an allergic reaction. They typically include a single protein source and a single carbohydrate source.\n\n2. Novel Protein Diets: These dog foods use proteins that are not commonly found in traditional dog foods, such as venison, duck, or salmon. This can help reduce the risk of an allergic reaction for dogs that have developed allergies to more common proteins like beef or chicken.\n\n3. Grain-Free Diets: Some dogs may have allergies to grains such as wheat, corn, or soy. Grain-free dog foods eliminate these ingredients, which can help alleviate symptoms of food allergies.\n\n4. Raw Diet: Some dog owners opt for a raw diet, which consists of uncooked meat, fruits, and vegetables. This diet can be beneficial for dogs with food allergies, as it eliminates the processing and preservatives found in commercial dog foods.\n\n5. Homemade Diet: Some dog owners choose to prepare a homemade diet for their dog, using ingredients that they know their dog is not allergic to. This can be a time-consuming and expensive option, but it allows for complete control over the ingredients in the dog's diet.\n\nIt's important to note that every dog is different, and what works for one dog may not work for another. If your dog has food allergies, it's best to work with your veterinarian to determine the best diet for their specific needs.",
                    },
                    {
                        "role": "user",
                        "content": "What are the best dog food options for a small breed dog like a Chihuahua?",
                    },
                ]
            }
        ]
    ],
}

# COMMAND ----------

response = score_model(dataset=dialog_test)
response

# COMMAND ----------

dialog_test = {
    "columns": ["messages"],
    "data": [
        [
            {
                "messages": [{'role': 'user',
  'content': 'How often should I feed my senior dog to prevent weight gain?'},
 {'role': 'assistant',
  'content': "  Once or twice a day. You can start with the recommended amount on the package and adjust accordingly to maintain your dog's weight. It's also important to consult with your veterinarian for specific feeding instructions tailored to your dog's individual needs."},
 {'role': 'user',
  'content': 'What are some hypoallergenic dog food options for a dog with food allergies?'},
 {'role': 'assistant',
  'content': "  There are several hypoallergenic dog food options available for dogs with food allergies. Some examples include:\n\n1. Limited Ingredient Diets (LIDs): These dog foods contain a minimal number of ingredients, which can help reduce the risk of an allergic reaction. They typically include a single protein source and a single carbohydrate source.\n\n2. Novel Protein Diets: These dog foods use proteins that are not commonly found in traditional dog foods, such as venison, duck, or salmon. This can help reduce the risk of an allergic reaction for dogs that have developed allergies to more common proteins like beef or chicken.\n\n3. Grain-Free Diets: Some dogs may have allergies to grains such as wheat, corn, or soy. Grain-free dog foods eliminate these ingredients, which can help alleviate symptoms of food allergies.\n\n4. Raw Diet: Some dog owners opt for a raw diet, which consists of uncooked meat, fruits, and vegetables. This diet can be beneficial for dogs with food allergies, as it eliminates the processing and preservatives found in commercial dog foods.\n\n5. Homemade Diet: Some dog owners choose to prepare a homemade diet for their dog, using ingredients that they know their dog is not allergic to. This can be a time-consuming and expensive option, but it allows for complete control over the ingredients in the dog's diet.\n\nIt's important to note that every dog is different, and what works for one dog may not work for another. If your dog has food allergies, it's best to work with your veterinarian to determine the best diet for their specific needs."},
 {'role': 'user',
  'content': 'What are the best dog food options for a small breed dog like a Chihuahua?'}]
            }
        ]
    ],
}

# COMMAND ----------

response = score_model(dataset=dialog_test)
response

# COMMAND ----------

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import re
import os

url = os.environ.get("ENDPOINT_URL")

# Define the asynchronous function to make API calls
async def llama(session, url, question, semaphore):
    async with semaphore:  # Acquire a spot in the semaphore
        # url = 'https://adb-6042569476650449.9.azuredatabricks.net/serving-endpoints/petm_chatbot_endpoint_main_databricks_petm_chatbot/invocations'
        headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
        dataset = get_response_data(question=question)
        data_json = {'dataframe_split': dataset}
        
        async with session.post(url=url, json=data_json, headers=headers) as response:
            return await response.json()

async def main(url, questions, max_concurrent_tasks):
    semaphore = asyncio.Semaphore(max_concurrent_tasks)  # Control concurrency
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(1000):  # Adjust the range as needed
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
