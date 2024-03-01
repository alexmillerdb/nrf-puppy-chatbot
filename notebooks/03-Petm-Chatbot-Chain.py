# Databricks notebook source
# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2 transformers==4.34.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md Helper function

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

# MAGIC %md ### Simple example of Langchain using ChatDatabricks

# COMMAND ----------

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer to this question: {question}"
)
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 500)

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

prompt_with_history_str = """
Your are a Pet Specialty retailer chatbot for dogs. Please answer Pet questions about dogs, dog services, and dog products only. If you don't know or not related to pets and dogs, don't answer.

Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)

# COMMAND ----------

# MAGIC %md ### Truncating Chat History length

# COMMAND ----------

import transformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

# COMMAND ----------

def convert_to_chat_hist(python_str):
    # Splitting the string by 'user:' and 'assistant:' while keeping these delimiters
    parts = []
    temp = python_str
    for delimiter in ["user:", "assistant:"]:
        if temp:
            split_text = temp.split(delimiter)
            first = split_text.pop(0)
            temp = delimiter.join(split_text)
            if first:
                parts.append(first)
            parts.extend([delimiter + s for s in split_text])

    # Removing empty strings if any
    parts = [part for part in parts if part.strip()]

    # Parsing each part into role and content
    messages = []
    for part in parts:
        if part.startswith("user:"):
            role = "user"
        elif part.startswith("assistant:"):
            role = "assistant"
        else:
            continue  # Skip if it doesn't start with a known role

        content = part[len(role) + 1:].strip()  # +1 for the colon after the role
        messages.append({"role": role, "content": content})

    return {"messages": messages}

# COMMAND ----------

def truncate_chat_history(chat_hist):
    # Function to tokenize and count tokens
    def tokenize_and_count(text):
        tokens = tokenizer.encode(text)
        return tokens, len(tokens)

    # Extract messages and concatenate them into one string
    # all_messages = ' '.join([msg['role'] + ": " + msg['content'] for msg in chat_hist['messages']])
    all_messages = ' '.join([msg['role'] + ": " + msg['content'] for msg in chat_hist])
    
    # Tokenize and count tokens in the entire chat
    tokens, total_count = tokenize_and_count(all_messages)
    # return total_count

    # If the total token count is more than 750, truncate
    if total_count > 500:
        truncated_tokens = tokens[-500:]  # Keep the last 750 tokens
        truncated_decode = tokenizer.decode(truncated_tokens)
        return convert_to_chat_hist(truncated_decode)
    else:
        return chat_hist

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

truncated_chat = truncate_chat_history(chat_hist['messages'])
print(truncated_chat)

# COMMAND ----------

from langchain.schema.runnable import RunnableLambda
from operator import itemgetter
from transformers import AutoTokenizer

# count tokens of history and return max of 500
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text))

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

# #The history is everything before the last question
# def extract_history(input):
#     return input[:-1]

def extract_history(input):
    return truncate_chat_history(input[:-1])
    

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
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

# MAGIC %md Add prompt templates to config file

# COMMAND ----------

# chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 200)

is_question_about_petsmart_str = """
You are classifying documents to know if this question is related with dogs, puppies, pet food, pet supplies, and other dog related items at Pet Specialty retailer. Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What is PetSmart?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: What is PetSmart?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_petsmart_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_petsmart_str
)

is_about_petsmart_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_petsmart_prompt
    | chat_model
    | StrOutputParser()
)

#Returns "Yes" as this is about PetSmart: 
print(is_about_petsmart_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a retail chain that specializes in pet supplies and services, such as food, toys, grooming, and training for dogs, cats, birds, fish, and other small animals."}, 
        {"role": "user", "content": "Does it include any Services like dog grooming?"}
    ]
}))

# COMMAND ----------

#Return "no" as this isn't about Databricks
print(is_about_petsmart_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}))

# COMMAND ----------

# MAGIC %md ### Use Vector Search Index to retrieve documents/products

# COMMAND ----------

# MAGIC %md ADD TO CONFIG

# COMMAND ----------

catalog = "main"
db = "databricks_petm_chatbot"
index_name=f"{catalog}.{db}.petm_data_embedded_index"
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
VECTOR_SEARCH_ENDPOINT_NAME = "petm_genai_chatbot"
text_column = "text"
vsc_columns = ["title", "url", "source"]

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
import os

# os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None, columns=vsc_columns):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model, columns=columns
    )
    return vectorstore.as_retriever(search_kwargs={'k': 3})

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
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

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
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

question_with_history_and_context_str = """
You are a trustful assistant for PetSmart customers. You are answering dog food preferences based on life stages, flavors, food type (dry vs. wet), brands, formulations, and more related to PetSmart's product catalog. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])
  
  # "item_title", "web_item_page_url"

def extract_source_urls(docs):
    return list(set([d.metadata["url"] for d in docs]))
  
def extract_source_titles(docs):
  return list(set([d.metadata["title"] for d in docs]))

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "titles": itemgetter("relevant_docs") | RunnableLambda(extract_source_titles),
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
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
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
# display_chat(non_relevant_dialog["messages"], response)

# COMMAND ----------

print(f'Testing with relevant history and question...')
response = full_chain.invoke(chat_hist)
response

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a Pet Specialty retailer."},  
        {"role": "user", "content": "Does PetSmart sell dog food for small breed dogs like chihuahuas?"},
        {"role": "assistant", "content": '  Yes, PetSmart does sell dog food for small breed dogs like Chihuahuas. Royal Canin Breed Health Nutrition Chihuahua Puppy Dry Dog Food is one example of dog food available at PetSmart that is specifically formulated for small breed dogs like Chihuahuas. Additionally, Canidae Pure Petite Small Breed All Life Stage Wet Dog Food is another option available at PetSmart that is designed for small breed dogs, including Chihuahuas.'},
        {"role": "user", "content": "I prefer Royal Canin brand, please recommend more of those items."},
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
response

# COMMAND ----------

# MAGIC %md ### Langchain documented steps/examples

# COMMAND ----------

chain_steps = full_chain.steps
chain_steps[0].invoke(dialog)

# COMMAND ----------

chain_steps[1]

# COMMAND ----------

# MAGIC %md ### Log to MLflow and UC

# COMMAND ----------

import cloudpickle
import pandas as pd
import mlflow
from mlflow.models import infer_signature
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.petm_chatbot_model"

with mlflow.start_run(run_name="petm_chatbot_rag") as run:
    #Get our model signature from input/output
    input_df = pd.DataFrame({"messages": [dialog]})
    output = full_chain.invoke(dialog)
    signature = infer_signature(input_df, output)

    model_info = mlflow.langchain.log_model(
        full_chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
            "pydantic==2.5.2 --no-binary pydantic",
            "cloudpickle=="+ cloudpickle.__version__,
            "transformers=="+ transformers.__version__
        ],
        input_example=input_df,
        signature=signature
    )

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(chat_hist)

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

display_chat(dialog["messages"], model.invoke(dialog))
