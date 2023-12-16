# Databricks notebook source
# %sh
# /databricks/python/bin/pip install filelock

# COMMAND ----------

# dbutils.fs.put("/databricks/init/install_filelock.sh", """
#     #!/bin/bash
#     /databricks/python/bin/pip install filelock
#     """, overwrite=True)
# spark.conf.set("spark.databricks.initScript.enabled", "true")
# spark.conf.set("spark.databricks.initScript.dbfs.file:///databricks/init/install_filelock.sh", "0")

# COMMAND ----------

# MAGIC %pip install transformers==4.30.2 langchain==0.0.319 llama-index==0.9.3 databricks-vectorsearch==0.20 pydantic==1.10.9 mlflow==2.9.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from langchain.text_splitter import TokenTextSplitter
import pandas as pd

# COMMAND ----------

catalog = "petsmart_chatbot"
schema = "datascience"

spark.sql(f'USE CATALOG {catalog}')
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# MAGIC %md ### Clean product catalog data:
# MAGIC - Remove HTML tags
# MAGIC - Concatenate item_name, flavor_desc, category_desc, health_consideration, and long_desc_cleaned
# MAGIC - Chunk text data

# COMMAND ----------

import pandas as pd
import re
import html

@F.pandas_udf(StringType())
def clean_text(text: pd.Series) -> pd.Series:
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

# COMMAND ----------

# product catalog data
product_data = spark.table("petm_product_catalog")

# use udf and concat_ws to concatenate the columns in `product_data`
product_data_cleaned = product_data \
    .withColumn("long_desc_cleansed", clean_text("long_desc")) \
    .withColumn("flavor_desc", F.when(F.col("flavor_desc").isNull(), F.lit("No flavor")).otherwise(F.col("flavor_desc"))) \
    .withColumn("flavor_desc_cleansed", clean_text(F.concat_ws(": ", F.lit("Flavor"), F.col("flavor_desc")))) \
    .withColumn("item_title_cleansed", clean_text(F.concat_ws(": ", F.lit("Item Title"), F.col("item_title")))) \
    .withColumn("category_desc_cleansed", F.concat_ws(": ", F.lit("Category Desc"), F.col("category_desc"))) \
    .withColumn("product_catalog_text", F.concat_ws("\n", *["item_title_cleansed", "flavor_desc_cleansed", "category_desc_cleansed", "long_desc_cleansed"])) \
    .withColumn("length_product_catalog_text", F.length("product_catalog_text"))

display(product_data_cleaned)

# COMMAND ----------

# MAGIC %md ### Chunking product catalog text data (DONT NEED FOR Product Catalog data)

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
  product_data_cleaned
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

catalog = "main"
# catalog = "hive_metastore"
schema = "databricks_petm_chatbot"
# schema = "default"

spark.sql(f'USE CATALOG {catalog}')
spark.sql(f'CREATE SCHEMA IF NOT EXISTS {schema}')
spark.sql(f"USE SCHEMA {schema}")

# COMMAND ----------

# product_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("main.databricks_petm_chatbot.petm_product_catalog_chunked")
product_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("petm_product_catalog_chunked")

# COMMAND ----------

# product_data_cleaned.write.format('delta').mode("overwrite").saveAsTable("hive_metastore.databricks_petm_chatbot.petm_product_catalog_cleansed")

# COMMAND ----------

# MAGIC %md ### Embed Product Catalog Data

# COMMAND ----------

from mlflow.deployments import get_deploy_client

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
print(embeddings)

# COMMAND ----------

# MAGIC %md ### Computing text embeddings and saving them to Delta

# COMMAND ----------

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

# COMMAND ----------

product_catalog_cleansed = spark.table("hive_metastore.databricks_petm_chatbot.petm_product_catalog_cleansed") \
    .withColumn()

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS product_data_embedded (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %md ### Dog blogs:
# MAGIC - Concatenate title and article
# MAGIC - Chunk articles and analyze edge cases

# COMMAND ----------

display(spark.table("dog_blogs")
        .withColumn("length", F.length("article")))

# COMMAND ----------

dog_blogs_df = spark.table("dog_blogs") \
  .withColumn("length", F.length("article")) \
  
  # .withColumn("title_article", F.concat(F.lit("Blog Title: "), F.col("title"), F.lit(" Article: ")))

dog_blogs_

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter

doc_text = "Protect Your Dog’s Dental Health: Doggie Breath or Something More?   As a pet parent, your dog’s health is always your top priority, but what about their dental health? Just like in humans, dogs need regular dental attention to ensure their teeth stay healthy. So what can you do to help keep their doggie smiles shiny and white? Believe it or not, proper dental health for dogs is quite similar to humans. Brushing your dog’s teeth and bringing them to the vet for regular dental cleanings will help promote better oral hygiene and keep their teeth healthier in the long run. Because dogs can develop periodontal disease (PD) without proper oral care, it’s very important that you give their teeth and gums regular attention. Learn more about how to keep your pup’s pearly whites healthy and what to be on the lookout for in relation to their oral health. Brush, Brush, Brush Your Dog's Teeth! When it comes to dental health for dogs, the most helpful tool for keeping plaque and tartar at bay is a simple toothbrush made especially for dogs. Daily brushing helps remove plaque, and therefore keeps tartar from forming. Using a toothbrush and toothpaste on your furry friend’s teeth is a great way to promote oral health for dogs. Your vet can help you choose which dog toothbrush and pet-safe toothpaste are best suited for your dog’s individual needs. You should also consult with your vet for tips and strategies for implementing brushing into your dog’s routine. Your pup can also help out when it comes to choosing the brush and paste they’ll be using. We have choices and flavors to satisfy everyone. And if any dog owners are nervous about the brushing process, they can always start off with a finger brush. Most dogs will eventually adapt and become comfortable with daily teeth cleaning, but starting your pet off with brushing as a puppy is recommended.   Regular Dental Visits at the Vet Having your pup’s smile regularly checked by a veterinarian is a preventive measure that should not be taken lightly. Taking your four-legged friend in for regular dental checkups and teeth cleanings can greatly reduce their risk for gum disease. The best way to make sure your dog’s oral health is on track is to have their teeth evaluated by a veterinarian, so any disease can be caught in the beginning stages.   Simply allowing your vet to have a look inside your pet’s mouth is enough to give them an idea of whether or not there are signs of dental disease. This is easy and painless for your pup, so no need to worry about them being in any discomfort. When your dog goes in for their scheduled teeth cleaning, your vet will also take an x-ray of their teeth to fully understand their dental health. During teeth cleanings, your dog will be put under general anesthesia while the vet does a thorough dental examination, teeth cleaning and polishing to remove the plaque and tartar from your dog’s teeth.    Dental Treats: Treat Your Dog to a Healthy Smile Feeding and treating your dog with the right types of treats and foods can be extremely helpful when it comes to dental health for dogs. There are some specialized foods and dental treats that are specifically formulated to promote healthy teeth and gums in canines.    If you’ve ever noticed dental treats in your local pet store, you’ve probably wondered whether they actually do any teeth cleaning. Studies have shown that dental treats are beneficial for the overall oral health of dogs. These specialized treats can help clean your four-legged friend’s teeth by removing plaque while they chew.    The simple decision to choose dental treats to reward your pup in place of traditional treats can have a lasting effect on oral health for dogs. Dental chews, bones, and biscuits are all great options to add to your dog’s daily routine to prevent tooth and gum issues.  When to Worry: Signs of Dental Disease in Dogs As we’ve covered, regular teeth cleaning, brushing, and use of dental treats and diets can significantly reduce the risk of dental disease and enhance oral health for dogs, but what if it’s too late? Poor dog dental care can, unfortunately, lead to serious complications in your dog’s mouth. According to an article published by VCA Hospitals, over 80% of dogs over the age of three have some sort of dental disease. And an estimated two-thirds of dogs suffer from periodontal disease, which means it’s the most common disease in dogs. So, as you may have guessed, it’s a good idea to keep an eye on your pup’s teeth and gums. Since they can’t effectively communicate that they’re in pain, staying on top of dog dental care is up to you as their pet parent.    Something as simple as halitosis (bad breath) can be a sign of dental disease. But how do you know if it’s just normal doggie breath, or something more serious? If you notice any of these symptoms, it could be a sign of dental disease in your dog:   - Bad Breath  - Discolored Teeth (yellow or brown) - Red or Swollen Gums - Receding Gums - Bleeding Gums - Loose Teeth   If your dog has any of the above signs and symptoms, be sure to make an appointment with your veterinarian right away for a proper diagnosis and a treatment plan. If dental disease worsens, it can result in surgical tooth extraction of loose or diseased teath.   Help your cuddly canine companion live their best life with PetSmart’s health and wellness products, including vitamins and supplements and other treatments. Shop online or stop by your nearest PetSmart today for everything you need to help your dog live a happy, healthy life.   Information in this article is not intended to diagnose, treat or cure your pet and is not a substitute for veterinary care provided by a veterinarian. For any medical or health-related advice concerning the care and treatment of your pet, contact your veterinarian.   References Hiscox, DVM, FAVD, Dip. AVDC, L., & Bellows, DVM, Dipl. AVDC, ABVP, J. (n.d.). Dental Disease in Dogs. VCA Hospitals. Retrieved 12 2, 2021, from https://vcahospitals.com/know-your-pet/dental-disease-in-dogs"

# COMMAND ----------

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == 0:
            self._index += 1
            return self.page_content
        raise StopIteration

# Your original text
# doc_text = "Your text goes here"

doc = Document(doc_text)
docs = [doc]  # Creating a list of Document instances

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(all_splits[0])
print(all_splits[1])

# COMMAND ----------

from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(all_splits[0])
print(all_splits[1])

# COMMAND ----------

display(spark.table("petm_faqs"))

# COMMAND ----------

from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

max_chunk_size = 1000
chunk_overlap = 100

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

# Split on H2, but merge small h2 chunks together to avoid too small. 
def split_html_on_h2(html, min_chunk_size = 20, max_chunk_size=500):
  h2_chunks = html_splitter.split_text(html)
  chunks = []
  previous_chunk = ""
  # Merge chunks together to add text before h2 and avoid too small docs.
  for c in h2_chunks:
    # Concat the h2 (note: we could remove the previous chunk to avoid duplicate h2)
    content = c.metadata.get('header2', "") + "\n" + c.page_content
    if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size/2:
        previous_chunk += content + "\n"
    else:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        previous_chunk = content + "\n"
  if previous_chunk:
      chunks.extend(text_splitter.split_text(previous_chunk.strip()))
  # Discard too small chunks
  return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]

# Let's create a user-defined function (UDF) to chunk all our documents with spark
@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
  
# Let's try our chunking function
html = spark.table("raw_documentation").limit(1).collect()[0]['text']
split_html_on_h2(html)
