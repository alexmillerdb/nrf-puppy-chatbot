# Databricks notebook source
# MAGIC %pip install transformers==4.30.2 langchain==0.0.319 llama-index==0.9.3 databricks-vectorsearch==0.20 pydantic==1.10.9 mlflow==2.9.0 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from langchain.text_splitter import TokenTextSplitter
import pandas as pd

# COMMAND ----------

dbutils.widgets.text("source_catalog", "petsmart_chatbot")
dbutils.widgets.text("source_schema", "datascience")

source_catalog = dbutils.widgets.get("source_catalog")
source_schema = dbutils.widgets.get("source_schema")

# COMMAND ----------

spark.sql(f'USE CATALOG {source_catalog}')
spark.sql(f"USE SCHEMA {source_schema}")

# COMMAND ----------

# MAGIC %md ### Clean product catalog data:
# MAGIC - Remove HTML tags
# MAGIC - Concatenate item_name, flavor_desc, category_desc, health_consideration, and long_desc_cleaned
# MAGIC - De-dupe products based on web style
# MAGIC - Chunk text data

# COMMAND ----------

from pyspark.sql import Window
from src.data_prep.utils import clean_text

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

# MAGIC %md ### Chunking product catalog text data (DONT NEED FOR Product Catalog data)

# COMMAND ----------

from src.data_prep.utils import ChunkData

# define tokenizer and chunk size/overlap
chunk_size = 1000
chunk_overlap = 150
tokenizer_name = "hf-internal-testing/llama-tokenizer"

# Create an instance of ChunkData
chunker = ChunkData(tokenizer_name=tokenizer_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Chunk data based on inputs
product_chunked_inputs = chunker.chunk_dataframe(product_data_ws, 'product_catalog_text')
display(product_chunked_inputs.select("item_id", "web_style_id", "long_desc_cleansed", "webimageurl", "flavor_desc_cleansed", 
                                      "item_title_cleansed", "category_desc_cleansed", "product_catalog_text", "chunks", "text"))

# COMMAND ----------

# MAGIC %md ### Write product data cleansed to UC

# COMMAND ----------

dbutils.widgets.text("dst_catalog", "main")
dbutils.widgets.text("dst_schema", "databricks_petm_chatbot")

dst_catalog = dbutils.widgets.get("dst_catalog")
dst_schema = dbutils.widgets.get("dst_schema")

# COMMAND ----------

spark.sql(f'USE CATALOG {dst_catalog}')
spark.sql(f'CREATE SCHEMA IF NOT EXISTS {dst_schema}')
spark.sql(f"USE SCHEMA {dst_schema}")

# COMMAND ----------

product_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("petm_web_style_chunked")

# COMMAND ----------

# MAGIC %md ### Embed Product Catalog Data

# COMMAND ----------

# MAGIC %md ### Computing text embeddings and saving them to Delta

# COMMAND ----------

from src.data_prep.utils import EmbeddingModel

embedding_model = EmbeddingModel(endpoint_name="databricks-bge-large-en")
df = spark.table("petm_web_style_chunked")
product_catalog_embedded = embedding_model.embed_text_data(df, "text") \
    .withColumn("id", F.monotonically_increasing_id()) \
    .select("item_id", "item_title", "web_style_id", "web_item_page_url", "webimageurl", "text", "embeddings") \
    .cache()

print(product_catalog_embedded.count())
display(product_catalog_embedded)

# COMMAND ----------

from src.data_prep.utils import create_cdc_table

create_cdc_table(table_name="web_style_data_embedded", df=product_catalog_embedded, spark=spark)
product_catalog_embedded.write.mode("overwrite").saveAsTable("web_style_data_embedded")
