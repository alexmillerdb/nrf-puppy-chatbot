# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch==0.20 mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ### Move to Config

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME = "petm_genai_chatbot"
catalog = "main"
db = schema = "databricks_petm_chatbot"
source_tables = ["web_style_data_embedded", "dog_blog_data_embedded", "faq_data_embedded"]
source_table = "petm_data_embedded"

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

def create_cdc_table(table_name, df):
    from delta import DeltaTable
    
    (DeltaTable.createIfNotExists(spark)
            .tableName(table_name)
            .addColumns(df.schema)
            .property("delta.enableChangeDataFeed", "true")
            .property("delta.columnMapping.mode", "name")
            .execute())

# COMMAND ----------

create_cdc_table(table_name="petm_data_embedded", df=final_df)
final_df.write.mode("overwrite").saveAsTable("petm_data_embedded")

# COMMAND ----------

# MAGIC %md ### Move to functions notebook

# COMMAND ----------

import time
def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status or i <6:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

def index_exists(vsc, endpoint_name, index_full_name):
    indexes = vsc.list_indexes(endpoint_name).get("vector_indexes", list())
    if any(index_full_name == index.get("name") for index in indexes):
      return True
    #Temp fix when index is not available in the list
    try:
        dict_vsindex = vsc.get_index(endpoint_name, index_full_name).describe()
        return dict_vsindex.get('status').get('ready')
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False
    
def wait_for_index_to_be_ready(vsc, vs_endpoint_name, index_name):
  for i in range(180):
    idx = vsc.get_index(vs_endpoint_name, index_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKNOWN'))
    if "ONLINE" in status:
      return
    if "UNKNOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return
    elif "PROVISIONING" in status:
      if i % 40 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

# MAGIC %md ### Create Vector Search Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md ### Create Vector Search Index

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c


#The table we'd like to index
source_table_fullname = f"{catalog}.{db}.{source_table}"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.{source_table}_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
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
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
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
