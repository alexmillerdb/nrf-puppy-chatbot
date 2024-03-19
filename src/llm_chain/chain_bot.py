# Databricks notebook source
# class ChatPreprocess:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
    
#     def convert_to_chat_hist(self, python_str):
#         # Splitting the string by 'user:' and 'assistant:' while keeping these delimiters
#         parts = []
#         temp = python_str
#         for delimiter in ["user:", "assistant:"]:
#             if temp:
#                 split_text = temp.split(delimiter)
#                 first = split_text.pop(0)
#                 temp = delimiter.join(split_text)
#                 if first:
#                     parts.append(first)
#                 parts.extend([delimiter + s for s in split_text])
#         # Removing empty strings if any
#         parts = [part for part in parts if part.strip()]
#         # Parsing each part into role and content
#         messages = []
#         for part in parts:
#             if part.startswith("user:"):
#                 role = "user"
#             elif part.startswith("assistant:"):
#                 role = "assistant"
#             else:
#                 continue  # Skip if it doesn't start with a known role
#             content = part[len(role) + 1:].strip()  # +1 for the colon after the role
#             messages.append({"role": role, "content": content})
#         return {"messages": messages}

#     def truncate_chat_history(self, chat_hist, token_count=500):
#         # Function to tokenize and count tokens
#         def tokenize_and_count(text):
#             tokens = self.tokenizer.encode(text)
#             return tokens, len(tokens)
#         # Extract messages and concatenate them into one string
#         all_messages = ' '.join([msg['role'] + ": " + msg['content'] for msg in chat_hist])
#         # Tokenize and count tokens in the entire chat
#         tokens, total_count = tokenize_and_count(all_messages)
#         # return total_count
#         # If the total token count is more than token_count, truncate
#         if total_count > token_count:
#             truncated_tokens = tokens[-token_count:]  # Keep the last token_count tokens
#             truncated_decode = self.tokenizer.decode(truncated_tokens)
#             return self.convert_to_chat_hist(truncated_decode)
#         else:
#             return chat_hist
        
#     def count_tokens(self, tokenizer, text):
#         return len(tokenizer.encode(text))

#     #The question is the last entry of the history
#     def extract_question(self, input):
#         return input[-1]["content"]

#     # Truncate history
#     def extract_history(self, input):
#         return self.truncate_chat_history(input[:-1])
    
#     def format_context(self, docs):
#         return "\n\n".join([d.page_content for d in docs])

#     def extract_source_urls(self, docs):
#         return list(set([d.metadata["url"] for d in docs]))
    
#     def extract_source_titles(self, docs):
#         return list(set([d.metadata["title"] for d in docs]))

# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatDatabricks
# from langchain.schema.output_parser import StrOutputParser

# prompt = PromptTemplate(
#   input_variables = ["question"],
#   template = "You are an assistant. Give a short answer to this question: {question}"
# )
# chat_model = ChatDatabricks(endpoint=chat_model_endpoint, max_tokens = chat_model_tokens)

# chain = (
#   prompt
#   | chat_model
#   | StrOutputParser()
# )

# import transformers
# from transformers import AutoTokenizer
# from langchain.schema.runnable import RunnableLambda
# from operator import itemgetter

# # load tokenizer
# tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

# # instantiate ChatPreprocess class
# chat_preprocess = ChatPreprocess(tokenizer=tokenizer)

# # create chain with history and add extract_question and extract_history methods
# chain_with_history = (
#     {
#         "question": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_question),
#         "chat_history": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_history),
#     }
#     | prompt_with_history
#     | chat_model
#     | StrOutputParser()
# )

# is_question_about_petsmart_prompt = PromptTemplate(
#   input_variables= ["chat_history", "question"],
#   template = is_question_about_petsmart_str
# )

# is_about_petsmart_chain = (
#     {
#         "question": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_question),
#         "chat_history": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_history),
#     }
#     | is_question_about_petsmart_prompt
#     | chat_model
#     | StrOutputParser()
# )

# from databricks.vector_search.client import VectorSearchClient
# from langchain.vectorstores import DatabricksVectorSearch
# from langchain.embeddings import DatabricksEmbeddings
# from langchain.chains import RetrievalQA
# import os

# # os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "rag_sp_token")
# os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
# os.environ['DATABRICKS_HOST'] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# def get_retriever(persist_dir: str = None):

#     # Get the vector search index
#     vsc = VectorSearchClient(workspace_url=os.environ["DATABRICKS_HOST"], 
#                              personal_access_token=os.environ['DATABRICKS_TOKEN'])
#     vs_index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, 
#                              index_name=index_name)

#     # Create the retriever
#     vectorstore = DatabricksVectorSearch(
#         vs_index, text_column=text_column, embedding=embedding_model, columns=vsc_columns
#     )
#     return vectorstore.as_retriever(search_kwargs={"k": 3})

# retriever = get_retriever()

# retrieve_document_chain = (
#     itemgetter("messages") 
#     | RunnableLambda(chat_preprocess.extract_question)
#     | retriever
# )

# from langchain.schema.runnable import RunnableBranch

# generate_query_to_retrieve_context_prompt = PromptTemplate(
#   input_variables= ["chat_history", "question"],
#   template = generate_query_to_retrieve_context_template
# )

# generate_query_to_retrieve_context_chain = (
#     {
#         "question": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_question),
#         "chat_history": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_history),
#     }
#     | RunnableBranch(  #Augment query only when there is a chat history
#       (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
#       (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
#       RunnableLambda(lambda x: x["question"])
#     )
# )

# from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough
# # from src.utils.llm_chain_utils import ChatPreprocess
# # from utils import ChatPreprocess

# question_with_history_and_context_prompt = PromptTemplate(
#   input_variables= ["chat_history", "context", "question"],
#   template = question_with_history_and_context_str
# )

# relevant_question_chain = (
#   RunnablePassthrough() |
#   {
#     "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
#     "chat_history": itemgetter("chat_history"), 
#     "question": itemgetter("question")
#   }
#   |
#   {
#     "context": itemgetter("relevant_docs") | RunnableLambda(chat_preprocess.format_context),
#     "sources": itemgetter("relevant_docs") | RunnableLambda(chat_preprocess.extract_source_urls),
#     "titles": itemgetter("relevant_docs") | RunnableLambda(chat_preprocess.extract_source_titles),
#     "chat_history": itemgetter("chat_history"), 
#     "question": itemgetter("question")
#   }
#   |
#   {
#     "prompt": question_with_history_and_context_prompt,
#     "sources": itemgetter("sources")
#   }
#   |
#   {
#     "result": itemgetter("prompt") | chat_model | StrOutputParser(),
#     "sources": itemgetter("sources")
#   }
# )

# irrelevant_question_chain = (
#   RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about PetSmart, dogs, or puppies.', "sources": []})
# )

# branch_node = RunnableBranch(
#   (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
#   (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
#   irrelevant_question_chain
# )

# full_chain = (
#   {
#     "question_is_relevant": is_about_petsmart_chain,
#     "question": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_question),
#     "chat_history": itemgetter("messages") | RunnableLambda(chat_preprocess.extract_history),    
#   }
#   | branch_node
# )

# import mlflow

# class PyfuncModel(mlflow.pyfunc.PythonModel):

#     def __init__(self, full_chain, retriever, tokenizer, databricks_host, databricks_token):
#         self.full_chain = full_chain
#         self.retriever = retriever
#         self.databricks_host = databricks_host
#         self.databricks_token = databricks_token
#         self.tokenizer = tokenizer

#     def load_context(self, context):

#         os.environ["DATABRICKS_HOST"] = self.databricks_host
#         os.environ["DATABRICKS_TOKEN"] = self.databricks_token
#         self.chat_process = ChatPreprocess(tokenizer=self.tokenizer)



#     def get_input(self, model_input):
#         import pandas as pd
#         import numpy as np

#         if isinstance(model_input, pd.DataFrame):
#             input_list = model_input.iloc[:, 0].tolist()
#         elif isinstance(model_input, np.ndarray):
#             input_list = model_input[:, 0].tolist()
#         else:
#             input_list = [model_input]
#         # elif isinstance(model_input, str):
#         #     input_list = [model_input]
#         # else: input_list = model_input

#         return input_list[0]

#     def predict(self, context, model_input):
#         # pass

#         input_field = self.get_input(model_input)

#         return self.full_chain.invoke(input_field)


# COMMAND ----------

# MAGIC %pip install mlflow==2.9.0 langchain==0.0.344 databricks-vectorsearch==0.22 cloudpickle==2.2.1 databricks-sdk==0.12.0 cloudpickle==2.2.1 pydantic==2.5.2 transformers==4.34.0 pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import mlflow
import langchain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnablePassthrough
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from transformers import AutoTokenizer
from operator import itemgetter
import yaml

class AppConfig:
    def __init__(self, config_path, section_list=["environment_config", "vector_search_config", "llm_chain_config", "chat_model_config"]):
        self.config_path = config_path
        self.databricks_token = os.environ['DATABRICKS_TOKEN']
        self.databricks_host = os.environ['DATABRICKS_HOST']
        self.load_config()
        self.process_sections(section_list)

    def load_config(self):
        """
        Load the YAML configuration file and set class attributes.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as yaml_file:
            self.config = yaml.safe_load(yaml_file)
        
        if not isinstance(self.config, dict):
            raise ValueError("Config is not in dict format")

    def process_sections(self, section_list):
        
        # Dynamically set attributes based on the YAML file
        # for key, value in self.config.items():
        for section in section_list:
            for key, value in self.config[section].items():
                setattr(self, key, value)

class ChatPreprocess:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def convert_to_chat_hist(self, python_str):
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

    def truncate_chat_history(self, chat_hist, token_count=500):
        # Function to tokenize and count tokens
        def tokenize_and_count(text):
            tokens = self.tokenizer.encode(text)
            return tokens, len(tokens)
        # Extract messages and concatenate them into one string
        all_messages = ' '.join([msg['role'] + ": " + msg['content'] for msg in chat_hist])
        # Tokenize and count tokens in the entire chat
        tokens, total_count = tokenize_and_count(all_messages)
        # return total_count
        # If the total token count is more than token_count, truncate
        if total_count > token_count:
            truncated_tokens = tokens[-token_count:]  # Keep the last token_count tokens
            truncated_decode = self.tokenizer.decode(truncated_tokens)
            return self.convert_to_chat_hist(truncated_decode)
        else:
            return chat_hist
        
    def count_tokens(self, tokenizer, text):
        return len(tokenizer.encode(text))
    
    def validate_input(self, input_data):
        if isinstance(input_data, dict) and 'messages' in input_data:
            return True
        return False
    
    def extract_question(self, input_data):
        if not self.validate_input(input_data):
            return "Invalid input format for extracting question."
        return input_data['messages'][-1]["content"]

    def extract_history(self, input_data):
        if not self.validate_input(input_data):
            return "Invalid input format for extracting history."
        history = input_data['messages'][:-1]
        return self.truncate_chat_history(history)

    # #The question is the last entry of the history
    # def extract_question(self, input):
    #     return input[-1]["content"]

    # # Truncate history
    # def extract_history(self, input):
    #     return self.truncate_chat_history(input[:-1])
    
    def format_context(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def extract_source_urls(self, docs):
        return list(set([d.metadata["url"] for d in docs]))
    
    def extract_source_titles(self, docs):
        return list(set([d.metadata["title"] for d in docs]))


class ChainFactory:
    def __init__(self, chat_preprocess, chat_model):
        self.chat_preprocess = chat_preprocess
        self.chat_model = chat_model
    # def __init__(self, chat_model_endpoint, chat_model_tokens, tokenizer):
        # self.chat_model = ChatDatabricks(endpoint=chat_model_endpoint, max_tokens=chat_model_tokens)
        # self.chat_preprocess = ChatPreprocess(tokenizer=tokenizer)

    def create_chain(self, prompt_template_str, input_vars):
        prompt = PromptTemplate(input_variables=input_vars, template=prompt_template_str)
        return (
            prompt
            | self.chat_model
            | StrOutputParser()
        )

    def create_chain_with_history(self, prompt_template_str, input_vars):
        prompt_with_history = PromptTemplate(input_variables=input_vars, template=prompt_template_str)
        return (
            {
                "question": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_question),
                "chat_history": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_history),
            }
            | prompt_with_history
            | self.chat_model
            | StrOutputParser()
        )

class RetrievalEmbeddingManager:
    def __init__(self, databricks_host, databricks_token, vector_search_endpoint_name, index_name):
        os.environ['DATABRICKS_TOKEN'] = databricks_token
        os.environ['DATABRICKS_HOST'] = databricks_host
        self.embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
        self.vsc = VectorSearchClient(workspace_url=databricks_host, personal_access_token=databricks_token)
        self.vs_index = self.vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=index_name)

    def get_retriever(self, text_column, vsc_columns, persist_dir=None):
        vectorstore = DatabricksVectorSearch(
            self.vs_index, text_column=text_column, embedding=self.embedding_model, columns=vsc_columns
        )
        return vectorstore.as_retriever(search_kwargs={"k": 3})


class PyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, config: AppConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_model)
        self.chat_preprocess = ChatPreprocess(self.tokenizer)
        self.chat_model = ChatDatabricks(endpoint=self.config.chat_model_endpoint, max_tokens=self.config.max_tokens)
        self.index_name = f"{self.config.catalog}.{self.config.db}.{self.config.vector_search_index}"
        self.retrieval_embedding_manager = RetrievalEmbeddingManager(self.config.databricks_host, self.config.databricks_token, self.config.vector_search_endpoint_name, self.index_name)
        # self.chain_factory = ChainFactory(self.chat_model, self.chat_preprocess)
        self.chain_factory = ChainFactory(self.chat_preprocess, self.chat_model)
        self.initialize_chains()

    def initialize_chains(self):
        # Define all your chains here
        self.chain = self.create_chain("You are an assistant. Give a short answer to this question: {question}", ["question"])
        self.chain_with_history = self.create_chain_with_history(self.config.prompt_with_history_str, ["chat_history", "question"])
        self.is_about_petsmart_chain = self.create_chain_with_history(self.config.is_question_about_petsmart_str, ["chat_history", "question"])
        self.retrieve_document_chain = self.create_retrieve_document_chain()
        self.generate_query_to_retrieve_context_chain = self.create_generate_query_to_retrieve_context_chain(self.config.generate_query_to_retrieve_context_template)
        self.relevant_question_chain = self.create_relevant_question_chain(self.config.question_with_history_and_context_str)
        self.irrelevant_question_chain = RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about PetSmart, dogs, or puppies.', "sources": []})
        self.branch_node = RunnableBranch(
            (lambda x: "yes" in x["question_is_relevant"].lower(), self.relevant_question_chain),
            (lambda x: "no" in x["question_is_relevant"].lower(), self.irrelevant_question_chain),
            self.irrelevant_question_chain
        )
        self.full_chain = (
            {
                "question_is_relevant": self.is_about_petsmart_chain,
                "question": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_question),
                "chat_history": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_history),    
            }
            | self.branch_node
        )

    def create_chain(self, prompt_template_str, input_vars):
        prompt = PromptTemplate(input_variables=input_vars, template=prompt_template_str)
        return (
            prompt
            | self.chat_model
            | StrOutputParser()
        )

    def create_chain_with_history(self, prompt_template_str, input_vars):
        prompt_with_history = PromptTemplate(input_variables=input_vars, template=prompt_template_str)
        return (
            {
                "question": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_question),
                "chat_history": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_history),
            }
            | prompt_with_history
            | self.chat_model
            | StrOutputParser()
        )

    def create_retrieve_document_chain(self):
        return (
            itemgetter("messages") 
            | RunnableLambda(self.chat_preprocess.extract_question)
            | self.retrieval_embedding_manager.get_retriever(text_column=self.config.text_column, vsc_columns=self.config.vsc_columns)
        )

    def create_generate_query_to_retrieve_context_chain(self, generate_query_to_retrieve_context_template):
        generate_query_to_retrieve_context_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template=generate_query_to_retrieve_context_template
        )
        return (
            {
                "question": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_question),
                "chat_history": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_history),
            }
            | RunnableBranch(  #Augment query only when there is a chat history
                (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | self.chat_model | StrOutputParser()),
                (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
                RunnableLambda(lambda x: x["question"])
            )
        )

    def create_relevant_question_chain(self, question_with_history_and_context_str):
        question_with_history_and_context_prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template=question_with_history_and_context_str
        )
        return (
            RunnablePassthrough() |
            {
                "relevant_docs": self.generate_query_to_retrieve_context_chain | self.chat_model | StrOutputParser() | 
                    self.retrieval_embedding_manager.get_retriever(text_column=self.config.text_column, vsc_columns=self.config.vsc_columns),
                "chat_history": itemgetter("chat_history"), 
                "question": itemgetter("question")
            }
            |
            {
                "context": itemgetter("relevant_docs") | RunnableLambda(self.chat_preprocess.format_context),
                "sources": itemgetter("relevant_docs") | RunnableLambda(self.chat_preprocess.extract_source_urls),
                "titles": itemgetter("relevant_docs") | RunnableLambda(self.chat_preprocess.extract_source_titles),
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
                "result": itemgetter("prompt") | self.chat_model | StrOutputParser(),
                "sources": itemgetter("sources")
            }
        )

    def load_context(self, context):
        # Set up environment variables and any other context setup
        os.environ["DATABRICKS_HOST"] = self.databricks_host
        os.environ["DATABRICKS_TOKEN"] = self.databricks_token

    # def get_input(self, model_input):
    #     import pandas as pd
    #     import numpy as np
    #     if isinstance(model_input, pd.DataFrame):
    #         input_list = model_input.iloc[:, 0].tolist()
    #     elif isinstance(model_input, np.ndarray):
    #         input_list = model_input[:, 0].tolist()
    #     else:
    #         input_list = [model_input]
    #     return input_list[0]

    def get_input(self, model_input):
        """
        Safely extract the 'messages' list from the model input.
        """
        # Handle dictionary input
        if isinstance(model_input, dict):
            # Directly return the 'messages' list if present
            return model_input.get("messages", [])

        # Handle DataFrame input
        elif isinstance(model_input, pd.DataFrame):
            # Attempt to extract a dict from the first cell and then the 'messages' list
            first_cell = model_input.iloc[0, 0]
            if isinstance(first_cell, dict):
                return first_cell.get("messages", [])
            else:
                raise ValueError("DataFrame does not contain dict with 'messages' in the first cell.")

        else:
            raise TypeError("Unsupported input type. Expected a dictionary or a DataFrame.")
    
    def predict(self, context, model_input):
        """
        Generate predictions based on the 'messages' content.
        """
        messages = self.get_input(model_input)  # Standardize and extract 'messages'
        if messages:
            return self.process_messages(messages)
        else:
            return "Input does not contain any messages."

    def process_messages(self, messages):
        """
        Processes the list of messages. Replace with actual logic.
        """
        # Here, you'd implement the actual processing logic.
        # For demonstration, we're directly invoking the full_chain's logic.
        # Ensure 'full_chain.invoke()' is correctly implemented to handle the 'messages' list.
        return self.full_chain.invoke({"messages": messages})

# COMMAND ----------

# Usage
target = "test"
config_path = f"../configs/{target}_config.yaml"
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
config = AppConfig(config_path)
model = PyfuncModel(config=config)

# COMMAND ----------

chat_preprocess = ChatPreprocess(model.tokenizer)
chat_preprocess.extract_history(non_relevant_dialog)

# COMMAND ----------

chain_factory = ChainFactory(chat_preprocess=chat_preprocess, chat_model=model.chat_model)

# COMMAND ----------

chat_hist = {
    "messages": [
            {"role": "user", "content": "What are the best food options for a puppy German Shepherd with sensitive skin?"},
            {"role": "assistant", "content": "  Based on the information provided, the best food options for a puppy German Shepherd with sensitive skin are Royal Canin Breed Health Nutrition German Shepherd Puppy Dry Dog Food and Hill's Prescription Diet Derm Complete Puppy Environmental/Food Sensitivities Dry Dog Food. Both foods are specifically formulated to support the skin's barriers against environmental irritants and provide nutritional support for building skin health. Additionally, they both contain ingredients that are easy to digest and are designed to promote healthy digestion. It's important to consult with a veterinarian to determine the best food option for your puppy's specific needs."},
            {"role": "user", "content": "How can I transition my adult dog from a puppy food to an adult food?"}]
}

# COMMAND ----------

prompt_with_history = chain_factory.create_chain_with_history(prompt_template_str=config.prompt_with_history_str, input_vars=["chat_history", "question"])
prompt_with_history.invoke(chat_hist)

# COMMAND ----------

def create_chain_with_history(self, prompt_template_str, input_vars):
        prompt_with_history = PromptTemplate(input_variables=input_vars, template=prompt_template_str)
        return (
            {
                "question": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_question),
                "chat_history": itemgetter("messages") | RunnableLambda(self.chat_preprocess.extract_history),
            }
            | prompt_with_history
            | self.chat_model
            | StrOutputParser()
        )

# COMMAND ----------

non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What is PetSmart?"}, 
        {"role": "assistant", "content": "PetSmart is a Pet Specialty retailer."}, 
        {"role": "user", "content": "Why is the sky blue?"}
    ]
}
model.full_chain.invoke(non_relevant_dialog)

# COMMAND ----------

model.predict("", non_relevant_dialog)

# COMMAND ----------

# Example dictionary input
dict_input = {
    "messages": [
        {"role": "user", "content": "What are the best food options for a puppy German Shepherd with sensitive skin?"},
        {"role": "assistant", "content": "Based on the information provided, the best food options..."},
        {"role": "user", "content": "How can I transition my adult dog from a puppy food to an adult food?"}
    ]
}

# Example DataFrame input
df_input = pd.DataFrame([dict_input])

# Predictions
# print(model.predict(None, dict_input))  # Passing None as context for simplicity
print(model.predict(None, df_input))

# COMMAND ----------

df_input.iloc[0, 0]

# COMMAND ----------

model.full_chain.invoke(dict_input)

# COMMAND ----------

non_relevant_dialog = {
    "messages": [
        {"role": "user", "content": "What is the meaning of life?"}
    ]
}
model.full_chain.invoke(non_relevant_dialog)

# COMMAND ----------

model.predict("", non_relevant_dialog)

# COMMAND ----------

chat_hist = {
    "messages": [
            {"role": "user", "content": "What are the best food options for a puppy German Shepherd with sensitive skin?"},
            {"role": "assistant", "content": "  Based on the information provided, the best food options for a puppy German Shepherd with sensitive skin are Royal Canin Breed Health Nutrition German Shepherd Puppy Dry Dog Food and Hill's Prescription Diet Derm Complete Puppy Environmental/Food Sensitivities Dry Dog Food. Both foods are specifically formulated to support the skin's barriers against environmental irritants and provide nutritional support for building skin health. Additionally, they both contain ingredients that are easy to digest and are designed to promote healthy digestion. It's important to consult with a veterinarian to determine the best food option for your puppy's specific needs."},
            {"role": "user", "content": "How can I transition my adult dog from a puppy food to an adult food?"}]
}

# COMMAND ----------

model.predict("", chat_hist)

# COMMAND ----------

import pandas as pd

# Assuming your chat_hist dictionary is as defined in the question
chat_hist_df = pd.DataFrame(chat_hist)  # Create a DataFrame with a single row

# Call the predict method with the DataFrame
result = model.predict("", chat_hist_df)

print(result)

# COMMAND ----------

model_input = chat_hist_df
for index, row in model_input.iterrows():
    # Assuming 'chat_hist' or the relevant data is in the first column
    chat_hist = row.iloc[0] if isinstance(row.iloc[0], dict) else row[0]
    messages = chat_hist.get("messages", [])

# COMMAND ----------

# model.predict("", chat_hist)
model.get_input(pd.DataFrame(chat_hist))
model.predict("", pd.DataFrame(chat_hist))

# COMMAND ----------

import cloudpickle
import pandas as pd
import mlflow
import transformers
from mlflow.pyfunc import PythonModel
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")
# model_name = f"{catalog}.{db}.{uc_model_name}"
model_name = f"{config.catalog}.{config.db}.{config.uc_model_name}"

with mlflow.start_run(run_name="petm_chatbot_rag_pyfunc") as run:

    #Get our model signature from input/output
    input_df = pd.DataFrame({"messages": [chat_hist]})
    output = model.predict("", input_df)
    signature = infer_signature(input_df, output)

    mlflow.log_dict(config, artifact_file=f"{target}_config.json")
    model_info = mlflow.pyfunc.log_model("chain", 
        registered_model_name=model_name,
        python_model=model, 
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

model.predict("", input_df)

# COMMAND ----------

model.get_input(input_df)

# COMMAND ----------

model.predict("", {"messages": [{
    "messages": 
        [
            {"role": "user", "content": "What are the best food options for a puppy German Shepherd with sensitive skin?"},
            {"role": "assistant", "content": "  Based on the information provided, the best food options for a puppy German Shepherd with sensitive skin are Royal Canin Breed Health Nutrition German Shepherd Puppy Dry Dog Food and Hill's Prescription Diet Derm Complete Puppy Environmental/Food Sensitivities Dry Dog Food. Both foods are specifically formulated to support the skin's barriers against environmental irritants and provide nutritional support for building skin health. Additionally, they both contain ingredients that are easy to digest and are designed to promote healthy digestion. It's important to consult with a veterinarian to determine the best food option for your puppy's specific needs."},
            {"role": "user", "content": "How can I transition my adult dog from a puppy food to an adult food?"}
            ]
}]})
