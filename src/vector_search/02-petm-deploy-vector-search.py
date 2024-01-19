# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch==0.20 mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### Define target/environment and import config file

# COMMAND ----------

dbutils.widgets.dropdown("target", "dev", ["dev", "staging", "prod"])
target = dbutils.widgets.get("target")

# COMMAND ----------

import json

with open(f"../configs/{target}_config.json", 'r') as cfg:
    config = json.load(cfg)

vs_config = config["vector_search_config"]
VECTOR_SEARCH_ENDPOINT_NAME = vs_config["VECTOR_SEARCH_ENDPOINT_NAME"]
catalog = vs_config["catalog"]
db = schema = vs_config["db"]
source_table = vs_config["source_table"]

print(f"Vector search endpoint name: {VECTOR_SEARCH_ENDPOINT_NAME}")
print(f"Catalog: {catalog}")
print(f"DB/Schema: {db}")
print(f"Source Table: {source_table}")

# COMMAND ----------

spark.sql(f'USE CATALOG {catalog}')
spark.sql(f'CREATE SCHEMA IF NOT EXISTS {schema}')
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md ### Unify all source tables into one dataframe:
# MAGIC - Creating 1 vector search index to embed into LLM chain

# COMMAND ----------

from pyspark.sql import functions as F

web_style_df = spark.table("web_style_data_embedded") \
    .select("text", "embeddings", F.col("web_item_page_url").alias("url"), F.col("item_title").alias("title")) \
    .withColumn("source", F.lit("product_catalog"))
    
display(web_style_df)

# COMMAND ----------

dog_blog_df = spark.table("dog_blog_data_embedded") \
    .select("text", "embeddings", F.col("url"), F.col("title")) \
    .withColumn("source", F.lit("blog"))

display(dog_blog_df)

# COMMAND ----------

faq_df = spark.table("faq_data_embedded") \
    .select("text", "embeddings", F.col("url"), F.col("question").alias("title"), F.col("context").alias("source")) \
    
display(faq_df)

# COMMAND ----------

final_df = web_style_df \
    .unionAll(dog_blog_df) \
    .unionAll(faq_df) \
    .withColumn("id", F.monotonically_increasing_id())

display(final_df)

# COMMAND ----------

from src.data_prep.utils import create_cdc_table

create_cdc_table(table_name="petm_data_embedded", df=final_df, spark=spark)
final_df.write.mode("overwrite").saveAsTable("petm_data_embedded")

# COMMAND ----------

# MAGIC %md ### Create Vector Search Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from src.vector_search.utils import VectorSearchUtility

vsc = VectorSearchClient()

if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

VectorSearchUtility.wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md ### Create Vector Search Index:
# MAGIC - `pipeline_type` can be "TRIGGERED" or "CONTINUOUS" 

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c


#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.{source_table}"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.{source_table}_index"

if not VectorSearchUtility.index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embeddings"
  )

#Let's wait for the index to be ready and all our embeddings to be created and indexed
VectorSearchUtility.wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md ### Search similar content from Vector Search Index

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "I have a German Shepherd adult who likes Chicken. I want to feed him a grain-free formulation, dry food."
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["title", "url", "source"],
  num_results=5)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

question = "What is the best way to potty train my dog?"
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["title", "url", "source"],
  num_results=5)
docs = results.get('result', {}).get('data_array', [])
docs
