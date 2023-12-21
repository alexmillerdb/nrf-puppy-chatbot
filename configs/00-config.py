# Databricks notebook source
# catalog for all UC assets (models, delta tables)
catalog = "main"
schema = db = "databricks_petm_chatbot"

# delta share catalog and schema
ds_catalog = "petsmart_chatbot"
ds_schema = "datascience"

# Vector Search Endpoint
VECTOR_SEARCH_ENDPOINT_NAME = "petm_genai_chatbot"
