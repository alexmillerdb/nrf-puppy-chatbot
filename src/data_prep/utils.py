import pandas as pd
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType, FloatType, IntegerType

# sc = SparkContext.getOrCreate()
# spark = SparkSession(sc)
# spark = SparkSession.builder.getOrCreate()
# def get_spark_session():
#     return SparkSession.builder.getOrCreate()

def create_cdc_table(table_name, df, spark):
    from delta import DeltaTable
    (DeltaTable.createIfNotExists(spark)
            .tableName(table_name)
            .addColumns(df.schema)
            .property("delta.enableChangeDataFeed", "true")
            .property("delta.columnMapping.mode", "name")
            .execute())


def clean_text(text: pd.Series) -> pd.Series:
    """
    Remove html tags, replace specific characters, and transform HTML character references in a string.

    Args:
        text (pd.Series): Input text data as Pandas Series.

    Returns:
        pd.Series: Output text data as Pandas Series.

    Example:
        >>> import pandas as pd
        >>> import re
        >>> import html
        >>> from pyspark.sql.functions import pandas_udf
        >>> from pyspark.sql.types import StringType
        >>> 
        >>> @F.pandas_udf(StringType())
        >>> def clean_text(text: pd.Series) -> pd.Series:
        >>>     def remove_html_replace_chars_transform_html_refs(s):
        >>>         if s is None:
        >>>             return s
        >>>         # Remove HTML tags
        >>>         clean_html = re.compile('<.*?>')
        >>>         s = re.sub(clean_html, '', s)
        >>>         # Replace specific characters
        >>>         s = s.replace("®", "")
        >>>         # Transform HTML character references
        >>>         s = html.unescape(s)
        >>>         # Additional logic for cases like 'dog#&39;s' -> 'dog 39s'
        >>>         s = re.sub(r'#&(\d+);', r' \1', s)
        >>>         return s
        >>>     return text.apply(remove_html_replace_chars_transform_html_refs)
        >>>
        >>> # Example usage
        >>> data = pd.Series(["A sample text <p>with HTML tags</p> and registered&reg; trademark", None, "Another text"])
        >>> clean_text(data)
        0              A sample text with HTML tags and registered trademark
        1                                                               None
        2                                                        Another text
        dtype: object
    """
    import pandas as pd
    import re
    import html
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import StringType

    @pandas_udf(StringType())
    def clean_text_udf(text: pd.Series) -> pd.Series:
        """Clean text from HTML tags, specific characters and HTML character references."""
        def remove_html_replace_chars_transform_html_refs(s):
            if s is None:
                return s
            clean_html = re.compile('<.*?>')
            s = re.sub(clean_html, '', s)
            s = s.replace("®", "")
            s = html.unescape(s)
            s = re.sub(r'#&(\d+);', r' \1', s)
            return s

        return text.apply(remove_html_replace_chars_transform_html_refs)

    return clean_text_udf(text)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

class ChunkData:
    def __init__(self, tokenizer_name="hf-internal-testing/llama-tokenizer", chunk_size=1000, chunk_overlap=150):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_chunks(self, text):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        return text_splitter.split_text(text.strip())

    def chunk_dataframe(self, df, text_column):

        @F.pandas_udf(ArrayType(StringType()))
        def chunker(docs):
            return docs.apply(self.get_chunks)
        return (df
            .withColumn("chunks", chunker(F.col(text_column)))  # Use col to reference the column
            .withColumn("num_chunks", F.size(F.col("chunks")))
            .withColumn("chunk", F.explode(F.col("chunks")))
            .withColumnRenamed("chunk", "text"))

import pandas as pd
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType
import mlflow.deployments

class EmbeddingModel:
    def __init__(self, endpoint_name="databricks-bge-large-en"):
        self.endpoint_name = endpoint_name

    def get_embeddings(self, batch):
        deploy_client = mlflow.deployments.get_deploy_client("databricks")
        response = deploy_client.predict(endpoint=self.endpoint_name, inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    def get_embedding(self, contents):
        max_batch_size = 150
        contents = pd.Series(contents)
        batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]
        all_embeddings = []
        for batch in batches:
            all_embeddings += self.get_embeddings(batch.tolist())
        return pd.Series(all_embeddings)

    def embed_text_data(self, df, text_column):
        @pandas_udf(ArrayType(FloatType()))
        def get_embedding_udf(series: pd.Series) -> pd.Series:
            return self.get_embedding(series)
        
        return df.withColumn("embeddings", get_embedding_udf(col(text_column)))
