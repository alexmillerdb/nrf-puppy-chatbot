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

# MAGIC %md ### Clean product catalog data:
# MAGIC - Remove HTML tags
# MAGIC - Concatenate item_name, flavor_desc, category_desc, health_consideration, and long_desc_cleaned
# MAGIC - De-dupe products based on web style
# MAGIC - Chunk text data

# COMMAND ----------
if update_data.lower() == "true":
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
        .filter(F.col("web_Style_rank") == 1)

    from src.data_prep.utils import ChunkData

    # Create an instance of ChunkData
    chunker = ChunkData(tokenizer_name=tokenizer_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Chunk data based on inputs
    product_chunked_inputs = chunker.chunk_dataframe(product_data_ws, 'product_catalog_text')

    dst_catalog = config["environment_config"]["catalog"]
    dst_schema = config["environment_config"]["schema"]

    spark.sql(f'USE CATALOG {dst_catalog}')
    spark.sql(f'CREATE SCHEMA IF NOT EXISTS {dst_schema}')
    spark.sql(f"USE SCHEMA {dst_schema}")

    product_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("petm_web_style_chunked")

    from src.data_prep.utils import EmbeddingModel

    embedding_model_endpoint = config["data_config"]["embedding_model_endpoint"]
    embedding_model = EmbeddingModel(endpoint_name=embedding_model_endpoint)
    df = spark.table("petm_web_style_chunked")
    product_catalog_embedded = embedding_model.embed_text_data(df, "text") \
        .withColumn("id", F.monotonically_increasing_id()) \
        .select("item_id", "item_title", "web_style_id", "web_item_page_url", "webimageurl", "text", "embeddings")

    from src.data_prep.utils import create_cdc_table
    from pyspark.sql import Window
    from src.data_prep.utils import clean_text

    create_cdc_table(table_name="web_style_data_embedded", df=product_catalog_embedded, spark=spark)
    product_catalog_embedded.write.mode("overwrite").saveAsTable("web_style_data_embedded")

else:
    print("Not updating code")
    dbutils.notebook.exit("Code not updated")