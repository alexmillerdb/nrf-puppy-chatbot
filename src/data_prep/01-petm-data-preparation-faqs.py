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

# MAGIC %md ### Clean dog blogs data:
# MAGIC - Remove HTML tags
# MAGIC - Chunk text data

# COMMAND ----------

from src.data_prep.utils import ChunkData

# define tokenizer and chunk size/overlap
chunk_size = 1000
chunk_overlap = 150
tokenizer_name = "hf-internal-testing/llama-tokenizer"

# Create an instance of ChunkData
chunker = ChunkData(tokenizer_name=tokenizer_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# chunk faqs data
faqs_df = spark.table("petm_faqs") \
  .withColumn("faq_context", F.concat(F.lit("Question: "), F.col("question"), F.lit(" Answer: "), F.col("answer"), F.lit(" context: "), F.col("context"))) \
  .withColumn("length", F.length("faq_context"))

faqs_df_chunked_inputs = chunker.chunk_dataframe(faqs_df, "faq_context").cache()
display(faqs_df_chunked_inputs)

# COMMAND ----------

# MAGIC %md ### Example of different text splitters (RecursiveCharacterTextSplitter vs. TokenTextSplitter)

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
    chunk_size=1000, chunk_overlap=150, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(all_splits[0])
print(all_splits[1])

# COMMAND ----------

from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=1000, chunk_overlap=150, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(all_splits[0])
print(all_splits[1])

# COMMAND ----------

# MAGIC %md ### Write chunked data to UC table

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

faqs_df_chunked_inputs.write.format("delta").mode("overwrite").saveAsTable("faqs_chunked")

# COMMAND ----------

# MAGIC %md ### Calculate Embeddings

# COMMAND ----------

# MAGIC %md ### Computing text embeddings and saving them to Delta

# COMMAND ----------

from src.data_prep.utils import EmbeddingModel

embedding_model = EmbeddingModel(endpoint_name="databricks-bge-large-en")
df = spark.table("faqs_chunked")
faqs_embedded = embedding_model.embed_text_data(df, "text") \
    .withColumn("id", F.monotonically_increasing_id()) \
    .cache()

display(faqs_embedded)

# COMMAND ----------

from src.data_prep.utils import create_cdc_table

create_cdc_table(table_name="faq_data_embedded", df=faqs_embedded, spark=spark)
faqs_embedded.write.mode("overwrite").saveAsTable("faq_data_embedded")

# COMMAND ----------

display(spark.table("faq_data_embedded"))
