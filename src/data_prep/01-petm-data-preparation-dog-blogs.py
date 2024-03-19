# Databricks notebook source
# MAGIC %pip install transformers==4.30.2 langchain==0.0.319 llama-index==0.9.3 databricks-vectorsearch==0.20 pydantic==1.10.9 mlflow==2.9.0 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from langchain.text_splitter import TokenTextSplitter
import pandas as pd
import yaml

from utils import load_config, create_cdc_table, EmbeddingModel

# COMMAND ----------

dbutils.widgets.text("update_data", "no")
dbutils.widgets.dropdown("target", "dev", ["dev", "staging", "prod"])

target = dbutils.widgets.get("target")
update_data = dbutils.widgets.get("update_data")

config_path = f"../configs/{target}_config.yaml"
config = load_config(config_path)
tokenizer_name = config["data_config"]["tokenizer_name"]
chunk_size = config["data_config"]["chunk_size"]
chunk_overlap = config["data_config"]["chunk_overlap"]
source_catalog = config["data_config"]["source_catalog"]
source_schema = config["data_config"]["source_schema"]

spark.sql(f'USE CATALOG {source_catalog}')
spark.sql(f"USE SCHEMA {source_schema}")

# COMMAND ----------

# MAGIC %md ### Clean dog blogs data:
# MAGIC - Remove HTML tags
# MAGIC - Chunk text data

# COMMAND ----------
if update_data.lower() == "true":
    # dog blogs from source table
    dog_blogs_df = spark.table("dog_blogs") \
      .withColumn("title_article", F.concat(F.lit("Blog Title: "), F.col("title"), F.lit(" Article: "), F.col("article"))) \
      .withColumn("length", F.length("title_article"))
    
    # Create an instance of ChunkData
    chunker = ChunkData(tokenizer_name=tokenizer_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Chunk data based on inputs
    dog_blogs_chunked_inputs = chunker.chunk_dataframe(dog_blogs_df, 'title_article')
    
    dst_catalog = config["environment_config"]["catalog"]
    dst_schema = config["environment_config"]["schema"]
    
    spark.sql(f'USE CATALOG {dst_catalog}')
    spark.sql(f'CREATE SCHEMA IF NOT EXISTS {dst_schema}')
    spark.sql(f"USE SCHEMA {dst_schema}")
    
    dog_blogs_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("dog_blogs_chunked")
    
    
    # load the embedding model
    embedding_model_endpoint = config["data_config"]["embedding_model_endpoint"]
    embedding_model = EmbeddingModel(endpoint_name=embedding_model_endpoint)
    
    df = spark.table("dog_blogs_chunked")
    dog_blogs_embedded = embedding_model.embed_text_data(df, "text") \
        .withColumn("id", F.monotonically_increasing_id()) \
        .cache()
    
    create_cdc_table(table_name="dog_blog_data_embedded", df=dog_blogs_embedded, spark=spark)
    dog_blogs_embedded.write.mode("overwrite").saveAsTable("dog_blog_data_embedded")
    
    display(spark.table("dog_blog_data_embedded"))
else:
    print("Update not needed")
    # Exit notebook
    dbutils.notebook.exit("Update not needed")
