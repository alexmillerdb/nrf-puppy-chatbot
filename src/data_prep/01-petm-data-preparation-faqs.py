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
    from src.data_prep.utils import ChunkData

    # Create an instance of ChunkData
    chunker = ChunkData(tokenizer_name=tokenizer_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # chunk faqs data
    faqs_df = spark.table("petm_faqs") \
    .withColumn("faq_context", F.concat(F.lit("Question: "), F.col("question"), F.lit(" Answer: "), F.col("answer"), F.lit(" context: "), F.col("context"))) \
    .withColumn("length", F.length("faq_context"))

    faqs_df_chunked_inputs = chunker.chunk_dataframe(faqs_df, "faq_context").cache()
    
    dst_catalog = config["environment_config"]["catalog"]
    dst_schema = config["environment_config"]["schema"]
    
    spark.sql(f'USE CATALOG {dst_catalog}')
    spark.sql(f'CREATE SCHEMA IF NOT EXISTS {dst_schema}')
    spark.sql(f"USE SCHEMA {dst_schema}")
    
    faqs_df_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("faqs_chunked")
    
    
    # load the embedding model
    embedding_model_endpoint = config["data_config"]["embedding_model_endpoint"]
    embedding_model = EmbeddingModel(endpoint_name=embedding_model_endpoint)
    
    df = spark.table("faqs_chunked")
    faqs_embedded = embedding_model.embed_text_data(df, "text") \
        .withColumn("id", F.monotonically_increasing_id()) \
        .cache()
    
    create_cdc_table(table_name="dog_blog_data_embedded", df=faqs_embedded, spark=spark)
    faqs_embedded.write.mode("overwrite").saveAsTable("faq_data_embedded")
    
    display(spark.table("faq_data_embedded"))
else:
    print("Update not needed")
    # Exit notebook
    dbutils.notebook.exit("Update not needed")
