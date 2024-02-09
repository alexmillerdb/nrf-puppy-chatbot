# Databricks notebook source
# MAGIC %pip install -r nrf-demo/requirements.txt

# COMMAND ----------

# MAGIC %pip install dbtunnel[ngrok,chainlit]==0.7.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os

current_directory = os.getcwd()
if str(current_directory).endswith("nrf-demo") is False:
    os.chdir("nrf-demo")
    current_directory = os.getcwd()
script_path = current_directory + "/main.py"
print(current_directory)
print(script_path)

# COMMAND ----------

from dbtunnel import dbtunnel

(dbtunnel.chainlit(script_path, cwd=current_directory)
.inject_env(
    DATABRICKS_HOST="<url with workspace model + indexes>",
    DATABRICKS_TOKEN="<token>" 
)   
.share_to_internet_via_ngrok(
    ngrok_api_token="<ngrok api token>",
    ngrok_tunnel_auth_token="<ngrok auth token>"
).run())
