# Databricks notebook source
# MAGIC %pip install textstat==0.7.3 tiktoken==0.5.1 evaluate==0.4.1 transformers==4.30.2 torch==1.13.1 "https://ml-team-public-read.s3.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_lakehouse_monitoring-0.4.4-py3-none-any.whl" mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME = "petm_genai_chatbot"
catalog = "main"
db = "databricks_petm_chatbot"
source_table = "web_style_data_embedded"
model_name = f"{catalog}.{db}.petm_chatbot_model"
serving_endpoint_name = f"petm_chatbot_endpoint_{catalog}_{db}"[:63]

# COMMAND ----------

# Set widgets for required parameters for this notebook.
dbutils.widgets.text("endpoint", f"{serving_endpoint_name}", label = "Name of Model Serving Endpoint")
endpoint_name = dbutils.widgets.get("endpoint")
if len(endpoint_name) == 0:
    raise Exception("Please fill in the required information for endpoint name.")


# Location to store streaming checkpoint
dbutils.widgets.text("checkpoint_location", f'dbfs:/Volumes/{catalog}/{db}/volume_petm_chatbot/checkpoints/payload_metrics', label = "Checkpoint Location")
checkpoint_location = dbutils.widgets.get("checkpoint_location")

# COMMAND ----------

import requests
from typing import Dict


def get_endpoint_status(endpoint_name: str) -> Dict:
    # Fetch the PAT token to send in the API request
    workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{workspace_url}/api/2.0/serving-endpoints/{endpoint_name}", json={"name": endpoint_name}, headers=headers).json()

    # Verify that Inference Tables is enabled.
    if "auto_capture_config" not in response.get("config", {}) or not response["config"]["auto_capture_config"]["enabled"]:
        raise Exception(f"Inference Tables is not enabled for endpoint {endpoint_name}. \n"
                        f"Received response: {response} from endpoint.\n"
                        "Please create an endpoint with Inference Tables enabled before running this notebook.")

    return response

response = get_endpoint_status(endpoint_name=endpoint_name)

auto_capture_config = response["config"]["auto_capture_config"]
catalog = auto_capture_config["catalog_name"]
schema = auto_capture_config["schema_name"]
# These values should not be changed - if they are, the monitor will not be accessible from the endpoint page.
payload_table_name = auto_capture_config["state"]["payload_table"]["name"]
payload_table_name = f"`{catalog}`.`{schema}`.`{payload_table_name}`"
print(f"Endpoint {endpoint_name} configured to log payload in table {payload_table_name}")

processed_table_name = f"{auto_capture_config['table_name_prefix']}_processed"
processed_table_name = f"`{catalog}`.`{schema}`.`{processed_table_name}`"
print(f"Processed requests with text evaluation metrics will be saved to: {processed_table_name}")

payloads = spark.table(payload_table_name).where('status_code == 200').limit(10)
display(payloads)

# COMMAND ----------

# MAGIC %md Define the Json Path to extract the input and output values

# COMMAND ----------

# The format of the input payloads, following the TF "inputs" serving format with a "query" field.
# Single query input format: {"inputs": [{"query": "User question?"}]}
# INPUT_REQUEST_JSON_PATH = "inputs[*].query"
# Matches the schema returned by the JSON selector (inputs[*].query is an array of string)
# INPUT_JSON_PATH_TYPE = "array<string>"
# KEEP_LAST_QUESTION_ONLY = False

# Example for format: {"dataframe_split": {"columns": ["messages"], "data": [[{"messages": [{"role": "user", "content": "What is Apache Spark?"}, {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, {"role": "user", "content": "Does it support streaming?"}]}]]}}
INPUT_REQUEST_JSON_PATH = "dataframe_split.data[0][*][*].messages[*].content"
INPUT_JSON_PATH_TYPE = "array<array<string>>"
# As we send in history, we only want to evaluate the last history input which is the new question.
KEEP_LAST_QUESTION_ONLY = True

# Answer format: {"predictions": ["answer"]}
#OUTPUT_REQUEST_JSON_PATH = "predictions"
# Matches the schema returned by the JSON selector (predictions is an array of string)
#OUPUT_JSON_PATH_TYPE = "array<string>"

# Answer format: {"predictions": [{"sources": ["https://docs"], "result": "  Yes."}]}
OUTPUT_REQUEST_JSON_PATH = "predictions[*].result"
# Matches the schema returned by the JSON selector (predictions is an array of string)
OUPUT_JSON_PATH_TYPE = "array<string>"

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import col, pandas_udf, transform, size, element_at

def unpack_requests(requests_raw: DataFrame, 
                    input_request_json_path: str, 
                    input_json_path_type: str, 
                    output_request_json_path: str, 
                    output_json_path_type: str,
                    keep_last_question_only: False) -> DataFrame:
    # Rename the date column and convert the timestamp milliseconds to TimestampType for downstream processing.
    requests_timestamped = (requests_raw
        .withColumnRenamed("date", "__db_date")
        .withColumn("__db_timestamp", (col("timestamp_ms") / 1000))
        .drop("timestamp_ms"))

    # Convert the model name and version columns into a model identifier column.
    requests_identified = requests_timestamped.withColumn(
        "__db_model_id",
        F.concat(
            col("request_metadata").getItem("model_name"),
            F.lit("_"),
            col("request_metadata").getItem("model_version")
        )
    )

    # Filter out the non-successful requests.
    requests_success = requests_identified.filter(col("status_code") == "200")

    # Unpack JSON.
    requests_unpacked = (requests_success
        .withColumn("request", F.from_json(F.expr(f"request:{input_request_json_path}"), input_json_path_type))
        .withColumn("response", F.from_json(F.expr(f"response:{output_request_json_path}"), output_json_path_type)))
    
    if keep_last_question_only:
        requests_unpacked = requests_unpacked.withColumn("request", transform(col("request"), lambda x: element_at(x, size(x))))

    # Explode batched requests into individual rows.
    requests_exploded = (requests_unpacked
        .withColumn("__db_request_response", F.explode(F.arrays_zip(col("request").alias("input"), col("response").alias("output"))))
        .selectExpr("* except(__db_request_response, request, response, request_metadata)", "__db_request_response.*")
        )

    return requests_exploded

# Let's try our unpacking function. Make sure input & output columns are not null
display(unpack_requests(payloads, INPUT_REQUEST_JSON_PATH, INPUT_JSON_PATH_TYPE, OUTPUT_REQUEST_JSON_PATH, OUPUT_JSON_PATH_TYPE, KEEP_LAST_QUESTION_ONLY))

# COMMAND ----------

# MAGIC %md ## Compute input/output text evaluation metrics such as toxicity, perplexity, readability

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import tiktoken, textstat, evaluate
import pandas as pd


@pandas_udf("int")
def compute_num_tokens(texts: pd.Series) -> pd.Series:
  encoding = tiktoken.get_encoding("cl100k_base")
  return pd.Series(map(len, encoding.encode_batch(texts)))

@pandas_udf("double")
def flesch_kincaid_grade(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.flesch_kincaid_grade(text) for text in texts])
 
@pandas_udf("double")
def automated_readability_index(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.automated_readability_index(text) for text in texts])

@pandas_udf("double")
def compute_toxicity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  toxicity = evaluate.load("toxicity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(toxicity.compute(predictions=texts.fillna(""))["toxicity"]).where(texts.notna(), None)

@pandas_udf("double")
def compute_perplexity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  perplexity = evaluate.load("perplexity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(perplexity.compute(data=texts.fillna(""), model_id="gpt2")["perplexities"]).where(texts.notna(), None)

# COMMAND ----------

from delta import DeltaTable

def compute_metrics(requests_df: DataFrame, column_to_measure = ["input", "output"]) -> DataFrame:
  for column_name in column_to_measure:
    requests_df = (
      requests_df.withColumn(f"toxicity({column_name})", compute_toxicity(F.col(column_name)))
                 .withColumn(f"perplexity({column_name})", compute_perplexity(F.col(column_name)))
                 .withColumn(f"token_count({column_name})", compute_num_tokens(F.col(column_name)))
                 .withColumn(f"flesch_kincaid_grade({column_name})", flesch_kincaid_grade(F.col(column_name)))
                 .withColumn(f"automated_readability_index({column_name})", automated_readability_index(F.col(column_name)))
    )
  return requests_df

# Initialize the processed requests table. Turn on CDF (for monitoring) and enable special characters in column names. 
def create_processed_table_if_not_exists(table_name, requests_with_metrics):
    (DeltaTable.createIfNotExists(spark)
        .tableName(table_name)
        .addColumns(requests_with_metrics.schema)
        .property("delta.enableChangeDataFeed", "true")
        .property("delta.columnMapping.mode", "name")
        .execute())

# COMMAND ----------

# MAGIC %md Incrementally consume new payload from inference table, unpack them, compute metrics and save them to final processed table

# COMMAND ----------

# %sql
# CREATE VOLUME IF NOT EXISTS main.databricks_petm_chatbot.volume_petm_chatbot;

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {processed_table_name}")

# COMMAND ----------

from delta.tables import DeltaTable

# Check whether the table exists before proceeding.
DeltaTable.forName(spark, payload_table_name)

# Unpack the requests as a stream.
requests_raw = spark.readStream.table(payload_table_name)
requests_processed = unpack_requests(requests_raw, INPUT_REQUEST_JSON_PATH, INPUT_JSON_PATH_TYPE, OUTPUT_REQUEST_JSON_PATH, OUPUT_JSON_PATH_TYPE, KEEP_LAST_QUESTION_ONLY)

# Drop columns that we don't need for monitoring analysis.
requests_processed = requests_processed.drop("date", "status_code", "sampling_fraction", "client_request_id", "databricks_request_id")

# Compute text evaluation metrics.
requests_with_metrics = compute_metrics(requests_processed)

# Persist the requests stream, with a defined checkpoint path for this table.
create_processed_table_if_not_exists(processed_table_name, requests_with_metrics)
(requests_with_metrics.writeStream
                      .trigger(availableNow=True)
                      .format("delta")
                      .outputMode("append")
                      .option("checkpointLocation", checkpoint_location)
                      .toTable(processed_table_name).awaitTermination())

# Display the table (with requests and text evaluation metrics) that will be monitored.
display(spark.table(processed_table_name))

# COMMAND ----------

# MAGIC %md ## Monitor inference table:
# MAGIC - Create monitor on inference table using `create_monitor` API. If monitor already exists, pass same parameters to `update_monitor`.

# COMMAND ----------

"""
Optional parameters to control monitoring analysis. For help, use the command help(lm.create_monitor).
"""
GRANULARITIES = ["1 day"]              # Window sizes to analyze data over
SLICING_EXPRS = None                   # Expressions to slice data with

CUSTOM_METRICS = None                  # A list of custom metrics to compute
BASELINE_TABLE = None                  # Baseline table name, if any, for computing baseline drift

# COMMAND ----------

import databricks.lakehouse_monitoring as lm


monitor_params = {
    "profile_type": lm.TimeSeries(
        timestamp_col="__db_timestamp",
        granularities=GRANULARITIES,
    ),
    "output_schema_name": f"{catalog}.{schema}",
    "schedule": None,  # We will refresh the metrics on-demand in this notebook
    "baseline_table_name": BASELINE_TABLE,
    "slicing_exprs": SLICING_EXPRS,
    "custom_metrics": CUSTOM_METRICS,
}

try:
    info = lm.create_monitor(table_name=processed_table_name, **monitor_params)
    print(info)
except Exception as e:
    # Ensure the exception was expected
    assert "RESOURCE_ALREADY_EXISTS" in str(e), f"Unexpected error: {e}"

    # Update the monitor if any parameters of this notebook have changed.
    lm.update_monitor(table_name=processed_table_name, updated_params=monitor_params)
    # Refresh metrics calculated on the requests table.
    refresh_info = lm.run_refresh(table_name=processed_table_name)
    print(refresh_info)

# COMMAND ----------

monitor = lm.get_monitor(table_name=processed_table_name)
url = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}/sql/dashboards/{monitor.dashboard_id}'
print(f"You can monitor the performance of your chatbot at {url}")

# COMMAND ----------

dbutils.notebook.exit(monitor)
