# Databricks notebook source
import os

# catalog for all UC assets (models, delta tables)
catalog = "main"
schema = db = "databricks_petm_chatbot"

# delta share catalog and schema
ds_catalog = "petsmart_chatbot"
ds_schema = "datascience"

# Vector Search Endpoint
VECTOR_SEARCH_ENDPOINT_NAME = "petm_genai_chatbot"
source_table = "petm_data_embedded"
vsc_columns = ["title", "url", "source"]
index_name=f"{catalog}.{db}.petm_data_embedded_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
text_column = "text"

# model config
max_tokens = 500
model_name = f"{catalog}.{db}.petm_chatbot_model"
serving_endpoint_name = f"petm_chatbot_endpoint_{catalog}_{db}"[:63]

# # environmental vars
# os.environ["DATABRICKS_TOKEN"] = (
#     dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# )

# os.environ["DATABRICKS_HOST"] = (
#     "https://" + spark.conf.get("spark.databricks.workspaceUrl")
# )

# os.environ["ENDPOINT_URL"] = os.path.join(
#     "https://" + spark.conf.get("spark.databricks.workspaceUrl"),
#     "serving-endpoints",
#     serving_endpoint_name,
#     "invocations",
# )
