import os
import requests
import time

from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
 
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

class EndpointApiClient:
    def __init__(self, base_url, token):
        # self.base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        # self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.base_url = base_url
        self.token = token
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