# Databricks notebook source
# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2 transformers==4.34.0 pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.dropdown("target", "dev", ["dev", "staging", "prod"])
target = dbutils.widgets.get("target")

# COMMAND ----------

# MAGIC %md Helper functions that are difficult to move to utils.py due to notebook dependencies

# COMMAND ----------

def display_chat(chat_history, response):
  def user_message_html(message):
    return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #c2efff; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px;">
        {message}
      </div>"""
  def assistant_message_html(message):
    return f"""
      <div style="width: 90%; border-radius: 10px; background-color: #e3f6fc; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
        <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/>
        {message}
      </div>"""
  chat_history_html = "".join([user_message_html(m["content"]) if m["role"] == "user" else assistant_message_html(m["content"]) for m in chat_history])
  answer = response["result"].replace('\n', '<br/>')
  sources_html = ("<br/><br/><br/><strong>Sources:</strong><br/> <ul>" + '\n'.join([f"""<li><a href="{s}">{s}</a></li>""" for s in response["sources"]]) + "</ul>") if response["sources"] else ""
  response_html = f"""{answer}{sources_html}"""

  displayHTML(chat_history_html + assistant_message_html(response_html))

# COMMAND ----------

# MAGIC %md ### Define configuration parameters

# COMMAND ----------

import yaml
import json

# Step 1: Load the YAML file
with open(f"../configs/{target}_config.yaml", 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)

assert (type(config) == dict), print("Config is not in dict format")

# COMMAND ----------

# environment config
env_config = config["environment_config"]
catalog = env_config["catalog"]
db = schema = env_config["db"]

# chat model config
chat_model_config = config["chat_model_config"]
chat_model_endpoint = chat_model_config["endpoint"]
chat_model_tokens = chat_model_config["max_tokens"]
uc_model_name = chat_model_config["uc_model_name"]

# llm chain config
llm_chain_config = config["llm_chain_config"]
prompt_with_history_str = llm_chain_config["prompt_with_history_str"]
is_question_about_petsmart_str = llm_chain_config["is_question_about_petsmart_str"]
generate_query_to_retrieve_context_template = llm_chain_config["generate_query_to_retrieve_context_template"]
question_with_history_and_context_str = llm_chain_config["question_with_history_and_context_str"]

# COMMAND ----------

# MAGIC %md ### Simple example of Langchain using ChatDatabricks

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer to this question: {question}"
)
chat_model = ChatDatabricks(endpoint=chat_model_endpoint, max_tokens = chat_model_tokens)

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
print(chain.invoke({"question": "What is PetSmart?"}))

# COMMAND ----------

# MAGIC %md ### Adding conversation to the history

# COMMAND ----------

# MAGIC %md Concepts to note:
# MAGIC - `RunnableLambda` runs custom functions and wraps it within a chain (https://python.langchain.com/docs/expression_language/how_to/functions)
# MAGIC - `RunnableParallel` useful for manipulating output of one Runnable to match the input format of next Runnable in sequence; eg map with keys "context" and "question" (https://python.langchain.com/docs/expression_language/how_to/map)
# MAGIC - `itemgetter` Python operation that returns a callable object that fetches item
# MAGIC - `RunnableBranch` dynamically routes logic based on input based on if conditions are met (https://python.langchain.com/docs/expression_language/how_to/routing)

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

chat_model = ChatDatabricks(endpoint=chat_model_endpoint, max_tokens = chat_model_tokens)

prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)

# COMMAND ----------

# MAGIC %md ### Truncating Chat History length

# COMMAND ----------

import transformers
from transformers import AutoTokenizer
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter
from src.utils.llm_chain_utils import ChatPreprocess
# from src.llm_chain.utils import ChatPreprocess
# from utils import ChatPreprocess

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

# Example usage
chat_hist = {
    "messages": [
                    {
                        "role": "user",
                        "content": "What are the best food options for a puppy German Shepherd with sensitive skin?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Based on the information provided, the best food options for a puppy German Shepherd with sensitive skin are Royal Canin Breed Health Nutrition German Shepherd Puppy Dry Dog Food and Hill's Prescription Diet Derm Complete Puppy Environmental/Food Sensitivities Dry Dog Food. Both foods are specifically formulated to support the skin's barriers against environmental irritants and provide nutritional support for building skin health. Additionally, they both contain ingredients that are easy to digest and are designed to promote healthy digestion. It's important to consult with a veterinarian to determine the best food option for your puppy's specific needs.",
                    },
                    {
                        "role": "user",
                        "content": "How can I transition my adult dog from a puppy food to an adult food?",
                    },
                    {
                        "role": "assistant",
                        "content": "  According to the information provided, it is recommended to gradually transition your adult dog from puppy food to adult food over a 7-10 day period. Replace 25% of the current diet with the new diet every 2-3 days until they are fully transitioned. It's also recommended to consult with your veterinarian for specific recommendations on transitioning your dog's diet and to ensure that the dog is healthy enough for the transition.",
                    },
                    {
                        "role": "user",
                        "content": "What are the best senior dog food options for a dog with joint issues?",
                    },
                    {
                        "role": "assistant",
                        "content": "Eukanuba Senior Small Breed Dry Dog Food and Eukanuba Senior Medium Breed Dry Dog Food are both good options for senior dogs with joint issues. They both contain glucosamine and chondroitin sulfate, which support joint health and mobility. Additionally, they have optimal levels of fats and carbohydrates to help maintain energy levels and support active lifestyles. They also provide immune system support and dental care.",
                    },
                    {
                        "role": "user",
                        "content": "My dog has a sensitive stomach, what are some limited ingredient dog food options?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Limited ingredient dog food options for sensitive stomachs include:\n\n1. Hill's Science Diet Sensitive Stomach & Skin Small Breed Adult Dry Dog Food - Chicken & Barley\n2. Purina Pro Plan Specialized Adult Dry Dog Food - Sensitive Skin & Stomach, Turkey & Oatmeal\n3. Purina Pro Plan Sensitive Skin & Stomach Adult Dry Dog Food - Salmon & Rice",
                    },
                    {
                        "role": "user",
                        "content": "What are the best dog treats for a puppy in training?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Wellness Soft Puppy Bites Natural Dog Treats - Natural, Lamb & Salmon.",
                    },
                    {
                        "role": "user",
                        "content": "Can you recommend a high-protein dog food for my active German Shepherd?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Sure! Based on your dog's activity level and breed size, I would recommend the Purina Pro Plan Sport All Life Stage Dry Dog Food - High Energy, High Protein, Salmon & Rice. It has a high protein content of 30% and 20% fat, which will support your dog's metabolic needs and help maintain lean, strong muscles. Additionally, it contains EPA and glucosamine for joint health and mobility, as well as guaranteed live probiotics for digestive and immune health. This recipe is designed for highly active dogs, including hunting and sporting dogs, and competitive canine athletes, making it a great fit for your German Shepherd.",
                    },
                    {
                        "role": "user",
                        "content": "What are the best toys for a puppy going through teething?",
                    },
                    {
                        "role": "assistant",
                        "content": "The best toys for a puppy going through teething are the Playology Puppy Teething Ball Dog Toy - Beef, the Playology Puppy Teething Ball Dog Toy - Peanut Butter, and the Playology Puppy Teething Ring Dog Toy - Peanut Butter. These toys are designed specifically for teething puppies and are made with natural rubber that is gentle on their gums. They also have unique textures and sounds that make them enticing for puppies to chew on, and they are built to last with puncture-resistant construction. Additionally, they use Playology's scent technology to drive 7 times more engagement than unscented toys, providing hours of stimulating playtime for teething puppies.",
                    },
                    {
                        "role": "user",
                        "content": "How often should I feed my senior dog to prevent weight gain?",
                    },
                    {
                        "role": "assistant",
                        "content": "  Once or twice a day. You can start with the recommended amount on the package and adjust accordingly to maintain your dog's weight. It's also important to consult with your veterinarian for specific feeding instructions tailored to your dog's individual needs.",
                    },
                    {
                        "role": "user",
                        "content": "What are some hypoallergenic dog food options for a dog with food allergies?",
                    },
                    {
                        "role": "assistant",
                        "content": "  There are several hypoallergenic dog food options available for dogs with food allergies. Some examples include:\n\n1. Limited Ingredient Diets (LIDs): These dog foods contain a minimal number of ingredients, which can help reduce the risk of an allergic reaction. They typically include a single protein source and a single carbohydrate source.\n\n2. Novel Protein Diets: These dog foods use proteins that are not commonly found in traditional dog foods, such as venison, duck, or salmon. This can help reduce the risk of an allergic reaction for dogs that have developed allergies to more common proteins like beef or chicken.\n\n3. Grain-Free Diets: Some dogs may have allergies to grains such as wheat, corn, or soy. Grain-free dog foods eliminate these ingredients, which can help alleviate symptoms of food allergies.\n\n4. Raw Diet: Some dog owners opt for a raw diet, which consists of uncooked meat, fruits, and vegetables. This diet can be beneficial for dogs with food allergies, as it eliminates the processing and preservatives found in commercial dog foods.\n\n5. Homemade Diet: Some dog owners choose to prepare a homemade diet for their dog, using ingredients that they know their dog is not allergic to. This can be a time-consuming and expensive option, but it allows for complete control over the ingredients in the dog's diet.\n\nIt's important to note that every dog is different, and what works for one dog may not work for another. If your dog has food allergies, it's best to work with your veterinarian to determine the best diet for their specific needs.",
                    },
                    {
                        "role": "user",
                        "content": "What are the best dog food options for a small breed dog like a Chihuahua?",
                    },
                ]
}

# instantiate ChatPreprocess class
chat_preprocess = ChatPreprocess(tokenizer=tokenizer)

# create chain with history and add extract_question and extract_history methods
chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_history),
    }
    | prompt_with_history
    | chat_model
    | StrOutputParser()
)

# Test that chat model is truncating chat messages
print(chain_with_history.invoke(chat_hist))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Let's add a filter on top to only answer Databricks-related questions.
# MAGIC
# MAGIC We want our chatbot to be profesionnal and only answer questions related to Databricks. Let's create a small chain and add a first classification step. 
# MAGIC
# MAGIC *Note: this is a fairly naive implementation, another solution could be adding a small classification model based on the question embedding, providing faster classification*

# COMMAND ----------

is_question_about_petsmart_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_petsmart_str
)

is_about_petsmart_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_history),
    }
    | is_question_about_petsmart_prompt
    | chat_model
    | StrOutputParser()
)

# COMMAND ----------

# MAGIC %md Unit testing to ensure `is_about_petsmart_chain` is working

# COMMAND ----------

assert (is_about_petsmart_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a retail chain that specializes in pet supplies and services, such as food, toys, grooming, and training for dogs, cats, birds, fish, and other small animals."}, 
        {"role": "user", "content": "Does it include any Services like dog grooming?"}
    ]
}).strip(" \n") == "Yes"), print(f"Prompt: {is_question_about_petsmart_prompt} is not working correctly")

assert (is_about_petsmart_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}).strip(" ") == "No"), print(f"Prompt: {is_question_about_petsmart_prompt} is not working correctly")

assert (is_about_petsmart_chain.invoke({
    "messages": [
        {"role": "user", "content": "Is the sky blue?"}
    ]
}).strip(" ") == "No"), print(f"Prompt: {is_question_about_petsmart_prompt} is not working correctly")

# COMMAND ----------

# MAGIC %md ### Use Vector Search Index to retrieve documents/products

# COMMAND ----------

# MAGIC %md Audit permissions using SP
# MAGIC - Instructions for setting up SP: https://docs.databricks.com/en/administration-guide/users-groups/service-principals.html

# COMMAND ----------

vs_config = config["vector_search_config"]
vs_src_table = vs_config["source_table"] + "_index"
index_name = f"{catalog}.{db}.{vs_src_table}"
VECTOR_SEARCH_ENDPOINT_NAME = vs_config["VECTOR_SEARCH_ENDPOINT_NAME"]
text_column = vs_config["text_column"]
vsc_columns = vs_config["vsc_columns"]

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
import os

# os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):

    # Get the vector search index
    vsc = VectorSearchClient(workspace_url=os.environ["DATABRICKS_HOST"], 
                             personal_access_token=os.environ['DATABRICKS_TOKEN'])
    vs_index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, 
                             index_name=index_name)

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column=text_column, embedding=embedding_model, columns=vsc_columns
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(chat_preprocess.extract_question)
    | retriever
)
print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "Does PetSmart sell dog food for sensitive stomach?"}]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improve document search using LLM to generate a better sentence for the vector store, based on the chat history
# MAGIC
# MAGIC We need to retrieve documents related the the last question but also the history.
# MAGIC
# MAGIC One solution is to add a step for our LLM to summarize the history and the last question, making it a better fit for our vector search query. Let's do that as a new step in our chain:

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

#Let's try it
output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the best dog food for my German Shepherd?"}
    ]
})
print(f"Test retriever query without history: {output}")
output = generate_query_to_retrieve_context_chain.invoke(chat_hist)
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough
# from src.utils.llm_chain_utils import ChatPreprocess
# from utils import ChatPreprocess

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(chat_preprocess.format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(chat_preprocess.extract_source_urls),
    "titles": itemgetter("relevant_docs") | RunnableLambda(chat_preprocess.extract_source_titles),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about PetSmart, dogs, or puppies.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_petsmart_chain,
    "question": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_history),    
  }
  | branch_node
)

# COMMAND ----------

import json
non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a Pet Specialty retailer."}, 
        {"role": "user", "content": "Why is the sky blue?"}
    ]
}

non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a Pet Specialty retailer."}, 
        {"role": "user", "content": "Who is the best puppy?"}
    ]
}
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
response

# COMMAND ----------

print(f'Testing with relevant history and question...')
response = full_chain.invoke(chat_hist)
response

# COMMAND ----------

# MAGIC %md ### Langchain documented steps/examples

# COMMAND ----------

chain_steps = full_chain.steps
chain_steps[0].invoke(chat_hist)

# COMMAND ----------

chain_steps[1]

# COMMAND ----------

# MAGIC %md ### Log to MLflow and UC:
# MAGIC - Log model as Pyfunc using custom class; this is needed to wrap src/utils.py code into model for model serving

# COMMAND ----------

import mlflow

class PyfuncModel(mlflow.pyfunc.PythonModel):

    def __init__(self, full_chain, retriever, tokenizer, databricks_host, databricks_token):
        self.full_chain = full_chain
        self.retriever = retriever
        self.databricks_host = databricks_host
        self.databricks_token = databricks_token
        self.tokenizer = tokenizer

    def load_context(self, context):

        os.environ["DATABRICKS_HOST"] = self.databricks_host
        os.environ["DATABRICKS_TOKEN"] = self.databricks_token
        self.chat_process = ChatPreprocess(tokenizer=self.tokenizer)



    def get_input(self, model_input):
        import pandas as pd
        import numpy as np

        if isinstance(model_input, pd.DataFrame):
            input_list = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, np.ndarray):
            input_list = model_input[:, 0].tolist()
        else:
            input_list = [model_input]
        # elif isinstance(model_input, str):
        #     input_list = [model_input]
        # else: input_list = model_input

        return input_list[0]

    def predict(self, context, model_input):
        # pass

        input_field = self.get_input(model_input)

        return self.full_chain.invoke(input_field)


# COMMAND ----------

chat_hist = {
    "messages": 
        [
            {"role": "user", "content": "What are the best food options for a puppy German Shepherd with sensitive skin?"},
            {"role": "assistant", "content": "  Based on the information provided, the best food options for a puppy German Shepherd with sensitive skin are Royal Canin Breed Health Nutrition German Shepherd Puppy Dry Dog Food and Hill's Prescription Diet Derm Complete Puppy Environmental/Food Sensitivities Dry Dog Food. Both foods are specifically formulated to support the skin's barriers against environmental irritants and provide nutritional support for building skin health. Additionally, they both contain ingredients that are easy to digest and are designed to promote healthy digestion. It's important to consult with a veterinarian to determine the best food option for your puppy's specific needs."},
            {"role": "user", "content": "How can I transition my adult dog from a puppy food to an adult food?"}
            ]
}

# COMMAND ----------

import pandas as pd
model_input = pd.DataFrame({"messages": [chat_hist]})

if isinstance(model_input, pd.DataFrame):
    input_list = model_input.iloc[:, 0][0]

input_list
full_chain.invoke(input_list)

# COMMAND ----------

python_model = PyfuncModel(full_chain=full_chain, retriever=retriever, databricks_host=os.environ["DATABRICKS_HOST"], databricks_token=os.environ["DATABRICKS_TOKEN"])

# COMMAND ----------

python_model.predict(context="", model_input=chat_hist)

# COMMAND ----------

import cloudpickle
import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models import infer_signature
import langchain

mlflow.set_registry_uri("databricks-uc")
# model_name = f"{catalog}.{db}.{uc_model_name}"
model_name = f"{catalog}.{db}.{uc_model_name}_pyfunc"

with mlflow.start_run(run_name="petm_chatbot_rag_pyfunc") as run:

    python_model = PyfuncModel(
        full_chain=full_chain, 
        retriever=retriever, 
        databricks_host=os.environ["DATABRICKS_HOST"], 
        databricks_token=os.environ["DATABRICKS_TOKEN"]
        )

    #Get our model signature from input/output
    input_df = pd.DataFrame({"messages": [chat_hist]})
    output = python_model.predict("", input_df)
    signature = infer_signature(input_df, output)
    # signature = infer_signature(input_df, output)

    mlflow.log_dict(config, artifact_file=f"{target}_config.json")
    model_info = mlflow.pyfunc.log_model("chain", 
        registered_model_name=model_name,
        python_model=python_model, 
        code_path = ["../utils/llm_chain_utils.py"],
        signature=signature, 
        input_example=input_df, 
        extra_pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch==0.22",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__,
            "transformers=="+ transformers.__version__
        ])

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_info.model_uri)

# COMMAND ----------

model.predict(input_df)

# COMMAND ----------

model.predict({"messages": [chat_hist]})

# COMMAND ----------

# import cloudpickle
# import pandas as pd
# import mlflow
# from mlflow.models import infer_signature
# import langchain

# mlflow.set_registry_uri("databricks-uc")
# model_name = f"{catalog}.{db}.{uc_model_name}"

# with mlflow.start_run(run_name="petm_chatbot_rag_pyfunc") as run:
#     #Get our model signature from input/output
#     input_df = pd.DataFrame({"messages": [chat_hist]})
#     output = full_chain.invoke(chat_hist)
#     signature = infer_signature(input_df, output)

#     mlflow.log_dict(config, artifact_file=f"{target}_config.json")
#     model_info = mlflow.langchain.log_model(
#         full_chain,
#         loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
#         artifact_path="chain",
#         registered_model_name=model_name,
#         pip_requirements=[
#             "mlflow==" + mlflow.__version__,
#             "langchain==" + langchain.__version__,
#             "databricks-vectorsearch",
#             "pydantic==2.5.2 --no-binary pydantic",
#             "cloudpickle=="+ cloudpickle.__version__,
#             "transformers=="+ transformers.__version__
#         ],
#         input_example=input_df,
#         signature=signature,
#         code_paths=["utils.py"]
#     )
#         # code_path=["./utils.py"]

# COMMAND ----------

# model = mlflow.langchain.load_model(model_info.model_uri)
# model.invoke(chat_hist)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a Pet Specialty retailer."},  
        {"role": "user", "content": "Does PetSmart have a Loyalty Program?"},
        {"role": "assistant", "content": "Yes, PetSmart has a loyalty program called Treats."},
        {"role": "user", "content": "How do I sign up for Treats?"}
    ]
}

# COMMAND ----------

# display_chat(dialog["messages"], model.invoke(dialog))
display_chat(dialog["messages"], model.predict({"messages": [dialog]}))
