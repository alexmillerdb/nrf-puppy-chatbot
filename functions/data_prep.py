from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

import pandas as pd

# create SparkSession
spark = SparkSession.builder.getOrCreate()

def use_catalog_schema(catalog, schema):
    """
    Uses a Catalog and Schema in Spark SQL.
    
    Args:
    - catalog (str): The name of the Catalog to use.
    - schema (str): The name of the Schema to use.
    
    Returns:
    - None
    
    Example:
    use_catalog_schema("my_catalog", "my_schema")
    """
    spark.sql(f'USE CATALOG {catalog}')
    spark.sql(f"USE SCHEMA {schema}")

@F.pandas_udf(StringType())
def clean_text(text: pd.Series) -> pd.Series:
    import pandas as pd
    import re
    import html
    
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

def create_cdc_table(table_name, df):
    from delta import DeltaTable
    
    (DeltaTable.createIfNotExists(spark)
            .tableName(table_name)
            .addColumns(df.schema)
            .property("delta.enableChangeDataFeed", "true")
            .property("delta.columnMapping.mode", "name")
            .execute())
