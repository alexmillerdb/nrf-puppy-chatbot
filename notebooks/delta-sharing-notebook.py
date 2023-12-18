# Databricks notebook source
catalog = "main"
schema = "databricks_petm_chatbot"

spark.sql(f'USE CATALOG {catalog}')
spark.sql(f'CREATE SCHEMA IF NOT EXISTS {schema}')
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

table_name = 'petm_product_catalog_chunked'
spark.sql(f"""
          CREATE TABLE IF NOT EXISTS {table_name}
          AS 
          SELECT * 
          FROM nrf_chatbot_am.databricks_petm_chatbot.{table_name}
          """)


# COMMAND ----------

display(spark.table(table_name))
