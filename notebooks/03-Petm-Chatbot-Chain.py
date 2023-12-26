# Databricks notebook source
# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../configs/00-config

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
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 1000)

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

from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | chat_model
    | StrOutputParser()
)

print(chain_with_history.invoke({
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a retail chain that specializes in pet supplies and services, such as food, toys, grooming, and training for dogs, cats, birds, fish, and other small animals."}, 
        {"role": "user", "content": "Does it include any Services like dog grooming?"}
    ]
}))

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

# max_tokens = 500
# chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = max_tokens)

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

# MAGIC %md Audit permissions using SP
# MAGIC - Instructions for setting up SP: https://docs.databricks.com/en/administration-guide/users-groups/service-principals.html

# COMMAND ----------

def test_demo_permissions(host, secret_scope, secret_key, vs_endpoint_name, index_name, embedding_endpoint_name = None):
  error = False
  CSS_REPORT = """
  <style>
  .dbdemos_install{
                      font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,Apple Color Emoji,Segoe UI Emoji,Segoe UI Symbol,Noto Color Emoji,FontAwesome;
  color: #3b3b3b;
  box-shadow: 0 .15rem 1.15rem 0 rgba(58,59,69,.15)!important;
  padding: 10px 20px 20px 20px;
  margin: 10px;
  font-size: 14px !important;
  }
  .dbdemos_block{
      display: block !important;
      width: 900px;
  }
  .code {
      padding: 5px;
      border: 1px solid #e4e4e4;
      font-family: monospace;
      background-color: #f5f5f5;
      margin: 5px 0px 0px 0px;
      display: inline;
  }
  </style>"""

  def display_error(title, error, color=""):
    displayHTML(f"""{CSS_REPORT}
      <div class="dbdemos_install">
                          <h1 style="color: #eb0707">Configuration error: {title}</h1> 
                            {error}
                        </div>""")
  
  def get_email():
    try:
      return spark.sql('select current_user() as user').collect()[0]['user']
    except:
      return 'Uknown'

  def get_token_error(msg, e):
    return f"""
    {msg}<br/><br/>
    Your model will be served using Databrick Serverless endpoint and needs a Pat Token to authenticate.<br/>
    <strong> This must be saved as a secret to be accessible when the model is deployed.</strong><br/><br/>
    Here is how you can add the Pat Token as a secret available within your notebook and for the model:
    <ul>
    <li>
      first, setup the Databricks CLI on your laptop or using this cluster terminal:
      <div class="code dbdemos_block">pip install databricks-cli</div>
    </li>
    <li> 
      Configure the CLI. You'll need your workspace URL and a PAT token from your profile page
      <div class="code dbdemos_block">databricks configure</div>
    </li>  
    <li>
      Create the dbdemos scope:
      <div class="code dbdemos_block">databricks secrets create-scope dbdemos</div>
    <li>
      Save your service principal secret. It will be used by the Model Endpoint to autenticate. <br/>
      If this is a demo/test, you can use one of your PAT token.
      <div class="code dbdemos_block">databricks secrets put-secret dbdemos rag_sp_token</div>
    </li>
    <li>
      Optional - if someone else created the scope, make sure they give you read access to the secret:
      <div class="code dbdemos_block">databricks secrets put-acl dbdemos '{get_email()}' READ</div>

    </li>  
    </ul>  
    <br/>
    Detailed error trying to access the secret:
      <div class="code dbdemos_block">{e}</div>"""

  try:
    secret = dbutils.secrets.get(secret_scope, secret_key)
    secret_principal = "__UNKNOWN__"
    try:
      from databricks.sdk import WorkspaceClient
      w = WorkspaceClient(token=dbutils.secrets.get(secret_scope, secret_key), host=host)
      secret_principal = w.current_user.me().emails[0].value
    except Exception as e_sp:
      error = True
      display_error(f"Couldn't get the SP identity using the Pat Token saved in your secret", 
                    get_token_error(f"<strong>This likely means that the Pat Token saved in your secret {secret_scope}/{secret_key} is incorrect or expired. Consider replacing it.</strong>", e_sp))
      return
  except Exception as e:
    error = True
    display_error(f"We couldn't access the Pat Token saved in the secret {secret_scope}/{secret_key}", 
                  get_token_error("<strong>This likely means your secret isn't set or not accessible for your user</strong>.", e))
    return
  
  try:
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=secret, disable_notice=True)
    vs_index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name)
    if embedding_endpoint_name:
      from langchain.embeddings import DatabricksEmbeddings
      embedding_model = DatabricksEmbeddings(endpoint=embedding_endpoint_name)
      embeddings = embedding_model.embed_query('What is Apache Spark?')
      results = vs_index.similarity_search(query_vector=embeddings, columns=["content"], num_results=1)
  except Exception as e:
    error = True
    vs_error = f"""
    Why are we getting this error?<br/>
    The model is using the Pat Token saved with the secret {secret_scope}/{secret_key} to access your vector search index '{index_name}' (host:{host}).<br/><br/>
    To do so, the principal owning the Pat Token must have USAGE permission on your schema and READ permission on the index.<br/>
    The principal is the one who generated the token you saved as secret: `{secret_principal}`. <br/>
    <i>Note: Production-grade deployement should to use a Service Principal ID instead.</i><br/>
    <br/>
    Here is how you can fix it:<br/><br/>
    <strong>Make sure your Service Principal has USE privileve on the schema</strong>:
    <div class="code dbdemos_block">
    spark.sql('GRANT USAGE ON CATALOG `{catalog}` TO `{secret_principal}>`');<br/>
    spark.sql('GRANT USAGE ON DATABASE `{catalog}`.`{db}` TO `{secret_principal}`');<br/>
    </div>
    <br/>
    <strong>Grant SELECT access to your SP to your index:</strong>
    <div class="code dbdemos_block">
    from databricks.sdk import WorkspaceClient<br/>
    import databricks.sdk.service.catalog as c<br/>
    WorkspaceClient().grants.update(c.SecurableType.TABLE, "{index_name}",<br/>
                                            changes=[c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="{secret_principal}")])
    </div>
    <br/>
    <strong>If this is still not working, make sure the value saved in your {secret_scope}/{secret_key} secret is your SP pat token </strong>.<br/>
    <i>Note: if you're using a shared demo workspace, please do not change the secret value if was set to a valid SP value by your admins.</i>

    <br/>
    <br/>
    Detailed error trying to access the endpoint:
    <div class="code dbdemos_block">{str(e)}</div>
    </div>
    """
    if "403" in str(e):
      display_error(f"Permission error on Vector Search index {index_name} using the endpoint {vs_endpoint_name} and secret {secret_scope}/{secret_key}", vs_error)
    else:
      display_error(f"Unkown error accessing the Vector Search index {index_name} using the endpoint {vs_endpoint_name} and secret {secret_scope}/{secret_key}", vs_error)
  def get_wid():
    try:
      return dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('orgId')
    except:
      return None
  if get_wid() in ["5206439413157315", "984752964297111", "1444828305810485", "2556758628403379"]:
    print(f"----------------------------\nYou are in a Shared FE workspace. Please don't override the secret value (it's set to the SP `{secret_principal}`).\n---------------------------")

  if not error:
    print('Secret and permissions seems to be properly setup, you can continue the demo!')

# COMMAND ----------

# MAGIC %md ADD TO CONFIG

# COMMAND ----------

# catalog = "main"
# db = "databricks_petm_chatbot"
# index_name=f"{catalog}.{db}.petm_data_embedded_index"
# host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
# VECTOR_SEARCH_ENDPOINT_NAME = "petm_genai_chatbot"
# text_column = "text"
# vsc_columns = ["title", "url", "source"]

# # #Let's make sure the secret is properly setup and can access our vector search index. Check the quick-start demo for more guidance
# test_demo_permissions(host, secret_scope="nrf-petm-chatbot", secret_key="rag_sp_token", vs_endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name=index_name, embedding_endpoint_name="databricks-bge-large-en")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
import os

# os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")
os.environ["DATABRICKS_HOST"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
# vsc_columns = ["url", "source"]

def get_retriever(persist_dir: str = None, columns=vsc_columns):
    # os.environ["DATABRICKS_HOST"] = host
    host = os.environ.get("DATABRICKS_HOST")
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

output = generate_query_to_retrieve_context_chain.invoke({
    "messages": [
        {"role": "user", "content": "What is the best dog food for my German Shepherd?"},
        {"role": "assistant", "content": "Royal Canin offers different formulas for German Shepherds depending on dogs needs."},
        {"role": "user", "content": "What type of flavors does Royal Canin offer for German Shepherds?"}
    ]
})
print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a trustful assistant for PetSmart customers. You are answering dog food preferences based on life stages, flavors, food type (dry vs. wet), brands, formulations, and more related to PetSmart's product catalog. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question. Always answer the question with a complete response. Never answer with incomplete responses.

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
    return [d.metadata["url"] for d in docs]
  
def extract_source_titles(docs):
  return [d.metadata["title"] for d in docs]

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
print(f'Testing with a non relevant question...')
response = full_chain.invoke(non_relevant_dialog)
response
# display_chat(non_relevant_dialog["messages"], response)

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a Pet Specialty retailer."},  
        {"role": "user", "content": "Does PetSmart sell dog food for small breed dogs like chihuahuas?"}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
response

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a Pet Specialty retailer."},  
        {"role": "user", "content": "Does PetSmart sell dog food for small breed dogs like chihuahuas?"},
        {"role": "assistant", "content": '  Yes, PetSmart does sell dog food for small breed dogs like Chihuahuas. Royal Canin Breed Health Nutrition Chihuahua Puppy Dry Dog Food is one example of dog food available at PetSmart that is specifically formulated for small breed dogs like Chihuahuas. Additionally, Canidae Pure Petite Small Breed All Life Stage Wet Dog Food is another option available at PetSmart that is designed for small breed dogs, including Chihuahuas.'},
        {"role": "user", "content": "I prefer Royal Canin brand, please recommend more of those items."}
    ]
}
print(f'Testing with relevant history and question...')
response = full_chain.invoke(dialog)
response

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "How often should I bathe my puppy, and what shampoos and conditioners are safe to use?"},
        # {"role": "assistant", "content": "  The frequency of bathing a puppy and the choice of shampoos and conditioners depend on several factors, such as the puppy's age, breed, health, and lifestyle. Generally, it is recommended to bathe a puppy every 4-6 weeks, unless they get dirty or develop a strong doggy odor.\n\nAs for shampoos and conditioners, it's important to use products specifically formulated for puppies' sensitive skin and coat. Look for products that are gentle, tearless, and pH-balanced. Avoid using human shampoos or conditioners on puppies, as they can be too harsh and cause irritation.\n\nTwo safe and effective options for puppy shampoos are the Only Natural Pet 2-in-1 Puppy Shampoo and the earthbath Ultra-Mild Puppy Sh"},
        # {"role": "user", "content": "The result that was provided was incomplete. Please regenerate but with complete sentences."} 
    ]
}

# COMMAND ----------

full_chain.invoke(dialog)

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

    mlflow.log_param("prompt_with_history_str", prompt_with_history_str)
    mlflow.log_param("is_question_about_petsmart_str", is_question_about_petsmart_str)
    mlflow.log_param("generate_query_to_retrieve_context_template", generate_query_to_retrieve_context_template)
    mlflow.log_param("question_with_history_and_context_str", question_with_history_and_context_str)
    

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
            "cloudpickle=="+ cloudpickle.__version__
        ],
        input_example=input_df,
        signature=signature
    )

# COMMAND ----------

model = mlflow.langchain.load_model(model_info.model_uri)
model.invoke(dialog)

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

# COMMAND ----------

dialog = {
    "messages": [
        {"role": "user", "content": "What are some best practices for potty training my puppy?"}, 
        {"role": "assistant", "content": "Pick a feeding schedule and adhere to it. Take your dog outside to the bathroom at the same spot each time.Reward your dog with treats and praise when they go to the bathroom outside. Keep an eye out for problems. Get the right supplies, such as a crate, training pads, and pet-specific cleaning supplies. Be consistent in everything. Stick to a schedule."},
        {"role": "user", "content": "Does PetSmart sell puppy pads?"}
    ]
}

# COMMAND ----------

display_chat(dialog["messages"], model.invoke(dialog))
