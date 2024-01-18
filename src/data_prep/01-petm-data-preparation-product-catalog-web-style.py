# Databricks notebook source
# MAGIC %pip install transformers==4.30.2 langchain==0.0.319 llama-index==0.9.3 databricks-vectorsearch==0.20 pydantic==1.10.9 mlflow==2.9.0 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from langchain.text_splitter import TokenTextSplitter
import pandas as pd

# COMMAND ----------

catalog = "petsmart_chatbot"
schema = "datascience"

spark.sql(f'USE CATALOG {catalog}')
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md ### Clean product catalog data:
# MAGIC - Remove HTML tags
# MAGIC - Concatenate item_name, flavor_desc, category_desc, health_consideration, and long_desc_cleaned
# MAGIC - Chunk text data

# COMMAND ----------

import pandas as pd
import re
import html

@F.pandas_udf(StringType())
def clean_text(text: pd.Series) -> pd.Series:
    """Remove html tags, replace specific characters, and transform HTML character references in a string"""
    def remove_html_replace_chars_transform_html_refs(s):
        if s is None:
            return s
        # Remove HTML tags
        clean_html = re.compile('<.*?>')
        s = re.sub(clean_html, '', s)
        # Replace specific characters
        s = s.replace("®", "")
        # Transform HTML character references
        s = html.unescape(s)
        # Additional logic for cases like 'dog#&39;s' -> 'dog 39s'
        s = re.sub(r'#&(\d+);', r' \1', s)
        return s

    return text.apply(remove_html_replace_chars_transform_html_refs)

# COMMAND ----------

from pyspark.sql import Window

# product catalog data
product_data = spark.table("petm_product_catalog")

# use udf and concat_ws to concatenate the columns in `product_data`
product_data_ws = product_data \
    .withColumn("long_desc_cleansed", clean_text("long_desc")) \
    .withColumn("flavor_desc", F.when(F.col("flavor_desc").isNull(), F.lit("No flavor")).otherwise(F.col("flavor_desc"))) \
    .withColumn("flavor_desc_cleansed", clean_text(F.concat_ws(": ", F.lit("Flavor"), F.col("flavor_desc")))) \
    .withColumn("item_title_cleansed", clean_text(F.concat_ws(": ", F.lit("Item Title"), F.col("item_title")))) \
    .withColumn("category_desc_cleansed", F.concat_ws(": ", F.lit("Category Desc"), F.col("category_desc"))) \
    .withColumn("product_catalog_text", F.concat_ws("\n", *["item_title_cleansed", "flavor_desc_cleansed", "category_desc_cleansed", "long_desc_cleansed"])) \
    .withColumn("length_product_catalog_text", F.length("product_catalog_text")) \
    .withColumn("web_style_rank", F.rank().over(Window.partitionBy("web_style_id").orderBy("item_id"))) \
    .filter(F.col("web_Style_rank") == 1) \
    .cache()

print(product_data_ws.count())
display(product_data_ws)

# COMMAND ----------

# MAGIC %md ### De-duping web style ID

# COMMAND ----------

# from pyspark.sql import Window

# cols = ['item_id', 'web_style_id', 'item_title', 'web_item_page_url', 'brand_name', 'size_desc', 'color_desc', 'flavor_desc', 'flavor_display_desc', 'category_desc', 'special_category_desc', 'channel_category_desc', 'health_category_desc', 'nutri_category_desc', 'pharmacy_item_ind', 'pharmacy_item_type', 'health_consideration', 'long_desc', 'short_desc', 'webimageurl', 'long_desc_cleansed', 'flavor_desc_cleansed', 'item_title_cleansed', 'category_desc_cleansed', 'product_catalog_text', 'length_product_catalog_text']

# # product_data_ws = spark.table("petm_product_catalog_chunked") \
# product_data_ws = product_data_cleaned \
#   .select(*cols).distinct() \
#   .withColumn("web_style_rank", F.rank().over(Window.partitionBy("web_style_id").orderBy("item_id"))) \
#   .filter(F.col("web_Style_rank") == 1) \
#   .cache()

# print(product_data_ws.count())
# display(product_data_ws)

# COMMAND ----------

# MAGIC %md ### Chunking product catalog text data (DONT NEED FOR Product Catalog data)

# COMMAND ----------

from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

chunk_size = 1000
chunk_overlap = 150

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

# @F.udf('array<string>')
def get_chunks(text):
 
  # instantiate tokenization utilities
  # text_splitter = TokenTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  
  # split text into chunks
  return text_splitter.split_text(text.strip())

@F.pandas_udf("array<string>")
def chunker(docs: pd.Series) -> pd.Series:
  return docs.apply(get_chunks)

# split text into chunks
product_chunked_inputs = (
  product_data_ws
    .withColumn('chunks', chunker('product_catalog_text')) # divide text into chunks
    .withColumn('num_chunks', F.expr("size(chunks)"))
    .withColumn('chunk', F.expr("explode(chunks)"))
    .withColumnRenamed('chunk','text')
  )

display(product_chunked_inputs.select("item_id", "web_style_id", "long_desc_cleansed", "webimageurl", "flavor_desc_cleansed", 
                                      "item_title_cleansed", "category_desc_cleansed", "product_catalog_text", "chunks", "text"))

# COMMAND ----------

# MAGIC %md ### Write product data cleansed to UC

# COMMAND ----------

catalog = "main"
schema = "databricks_petm_chatbot"

spark.sql(f'USE CATALOG {catalog}')
spark.sql(f'CREATE SCHEMA IF NOT EXISTS {schema}')
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# product_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("main.databricks_petm_chatbot.petm_product_catalog_chunked")
product_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("petm_web_style_chunked")

# COMMAND ----------

# MAGIC %md ### Embed Product Catalog Data

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
print(embeddings)

# COMMAND ----------

# MAGIC %md ### Computing text embeddings and saving them to Delta

# COMMAND ----------

@F.pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

product_catalog_embedded = spark.table("petm_web_style_chunked") \
    .withColumn("embeddings", get_embedding("text")) \
    .select("item_id", "item_title", "web_style_id", "web_item_page_url", "webimageurl", "text", "embeddings") \
    .withColumn("id", F.monotonically_increasing_id()) \
    .cache()

print(product_catalog_embedded.count())
display(product_catalog_embedded)

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS web_style_data_embedded (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   item_id INT,
# MAGIC   item_title STRING,
# MAGIC   web_style_id INT,
# MAGIC   web_item_page_url STRING,
# MAGIC   webimageurl STRING,
# MAGIC   text STRING,
# MAGIC   embeddings ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %md ### Need to update `overwrite` to add/append data

# COMMAND ----------

product_catalog_embedded.write.mode('overwrite').saveAsTable("web_style_data_embedded")

# COMMAND ----------

# MAGIC %md ### Create Self-Managed Vector Search Index (MOVE THIS TO SEPERATE NOTEBOOK TO INCLUDE 3 EMBEDDING TABLES)

# COMMAND ----------

# MAGIC %md Add to config

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME = "petm_genai_chatbot"
catalog = "main"
db = "databricks_petm_chatbot"
source_table = "web_style_data_embedded"

# COMMAND ----------

# MAGIC %md Helper functions

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

# MAGIC %md Create Vector Search Endpoint

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md Create Vector Search Index

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

catalog = "main"
db = "databricks_petm_chatbot"
source_table = "web_style_data_embedded"

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

# MAGIC %md ### Search for similar content

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "What is a good product for recommendation for Small Breed dog that has Sensitive Skin and Stomach for Purina Pro Plan?"
question = "I need a product recommendation for a large breed puppy for brands such as Pro Plan or Royal Canin."
question = "I have a German Shepherd adult who likes Chicken. I want to feed him a grain-free formulation, dry food."
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["item_title", "web_item_page_url"],
  num_results=5)
docs = results.get('result', {}).get('data_array', [])
docs
