# Databricks notebook source
# MAGIC %pip install transformers==4.30.2 langchain==0.0.319 llama-index==0.9.3 databricks-vectorsearch==0.20 pydantic==1.10.9 mlflow==2.9.0 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from langchain.text_splitter import TokenTextSplitter
import pandas as pd
from functions.data_prep import clean_text, get_embedding, create_cdc_table

# COMMAND ----------

# MAGIC %run ../configs/00-config

# COMMAND ----------

# Delta Share catalog and schema
spark.sql(f'USE CATALOG {ds_catalog}')
spark.sql(f"USE SCHEMA {ds_schema}")

# COMMAND ----------

# MAGIC %md ### Clean product catalog data:
# MAGIC - Remove HTML tags
# MAGIC - Concatenate item_name, flavor_desc, category_desc, health_consideration, and long_desc_cleaned
# MAGIC - Chunk text data

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

# MAGIC %md ### Chunking product catalog text data

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

spark.sql(f'USE CATALOG {catalog}')
spark.sql(f'CREATE SCHEMA IF NOT EXISTS {schema}')
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# product_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("main.databricks_petm_chatbot.petm_product_catalog_chunked")
product_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("petm_web_style_chunked")

# COMMAND ----------

# MAGIC %md ### Computing text embeddings and saving them to Delta

# COMMAND ----------

product_catalog_embedded = spark.table("petm_web_style_chunked") \
    .withColumn("embeddings", get_embedding("text")) \
    .select("item_id", "item_title", "web_style_id", "web_item_page_url", "webimageurl", "text", "embeddings") \
    .withColumn("id", F.monotonically_increasing_id()) \
    .cache()

print(product_catalog_embedded.count())
display(product_catalog_embedded)

# COMMAND ----------

create_cdc_table(table_name="web_style_data_embedded", df=product_catalog_embedded)
product_catalog_embedded.write.mode('overwrite').saveAsTable("web_style_data_embedded")

# COMMAND ----------

display(spark.table("web_style_data_embedded"))
