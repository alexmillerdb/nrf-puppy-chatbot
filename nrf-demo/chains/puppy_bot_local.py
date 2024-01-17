import asyncio
import copy
import logging
import time
from operator import itemgetter

from databricks.vector_search.client import VectorSearchClient
from databricks_genai_inference import ChatCompletion
from dotenv import load_dotenv
from langchain.chat_models import ChatDatabricks
from langchain.embeddings import DatabricksEmbeddings
from langchain.vectorstores.databricks_vector_search import DatabricksVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from chains import ChainlitChat, StreamingResponse, BatchResponse, AsyncGeneratorWrapper, HasMessage
import chainlit as cl

load_dotenv()

chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens=200)

prompt_with_history_str = """
Your are a Pet Specialty retailer chatbot for dogs. Please answer Pet questions about dogs, dog services, and dog products only. If you don't know or not related to pets and dogs, don't answer.

Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=prompt_with_history_str
)


def truncate_chat_history(input_) -> str:
    return input_


def extract_question(input_) -> str:
    return input_[-1]["content"]


def extract_history(input_) -> str:
    return truncate_chat_history(input_[:-1])


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
    input_variables=["chat_history", "question"],
    template=is_question_about_petsmart_str
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

vsc_columns = ["title", "url", "source"]


class VectorSearchTokenFilter(logging.Filter):
    def __init__(self, module_path):
        super().__init__()
        self.module_path = module_path

    @staticmethod
    def remove_bearer_content(msg):
        import re
        # Define a regex pattern to match 'Bearer ' followed by anything
        pattern = re.compile(r'\'Bearer\s.*\'')

        # Use re.sub to replace the matched pattern with 'Bearer '
        clean_token = re.sub(pattern, '\'Bearer *****\'', msg)

        return clean_token

    def filter(self, record):
        # Check if the record's module matches the specified module_name
        if record.pathname.endswith(self.module_path):
            record.msg = self.remove_bearer_content(record.msg)
        return True


def get_retriever(columns=None):
    custom_filter = VectorSearchTokenFilter("databricks/vector_search/utils.py")
    logging.getLogger().addFilter(custom_filter)
    columns = columns or vsc_columns
    # Get the vector search index
    vsc = VectorSearchClient()
    catalog = "main"
    db = "databricks_petm_chatbot"
    vector_search_endpoint_name = "petm_genai_chatbot"
    index_name = f"{catalog}.{db}.petm_data_embedded_index"
    embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
    vs_index = vsc.get_index(
        endpoint_name=vector_search_endpoint_name,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model, columns=columns
    )
    return vectorstore.as_retriever(search_kwargs={'k': 3})


retriever = get_retriever()

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=generate_query_to_retrieve_context_template
)

question_with_history_and_context_str = """
You are a trustful assistant for PetSmart customers. You are answering dog food preferences based on life stages, flavors, food type (dry vs. wet), brands, formulations, and more related to PetSmart's product catalog. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=question_with_history_and_context_str
)


def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])


def extract_source_urls(docs):
    return list(set([d.metadata["url"] for d in docs]))


def extract_source_titles(docs):
    return list(set([d.metadata["title"] for d in docs]))


relevant_question_chain_prompt = (
        RunnablePassthrough() |
        {
            "question": itemgetter("messages") | RunnableLambda(extract_question),
            "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
        } |
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
)


def run_chat_completion(msgs) -> AsyncGeneratorWrapper[HasMessage]:
    resp = ChatCompletion.create(model="llama-2-70b-chat",
                                 messages=[{"role": "system", "content": "You are a helpful assistant."},
                                           *msgs],
                                 temperature=0.1,
                                 stream=True, )
    return AsyncGeneratorWrapper(resp)


def get_history(input_):
    history = cl.user_session.get("history")
    if history is None:
        history = []
    history_copy = copy.deepcopy(history)
    history_copy.append({"role": "user", "content": f"{input_}"})
    # expected by petsmart lcel
    return {
        "messages": history_copy
    }


class PuppyBot(ChainlitChat):
    @staticmethod
    def _guard(content) -> str | None:
        start = time.time()
        is_valid_question = is_about_petsmart_chain.invoke(content).strip()
        print(f"[INFO] Time to compute guard: {time.time() - start}")
        print(f"[INFO] Is valid prompt: {content}; Guard Response: {is_valid_question}")
        if is_valid_question.lower().startswith("no"):
            print(f"[INFO] Invalid Question: Prompt: {content} Guard Response: {is_valid_question}")
            return "I am sorry I am not able to answer that question." + \
                " Please feel free to ask me questions about the various data, " + \
                "technology and service partners in the Databricks partner ecosystem."
        return None

    def intro_message(self) -> cl.Message | None:
        return cl.Message("Welcome to the Databricks NRF Puppy Chat Bot in collaboration with "
                          "PetSmart! Ask me a question about how to best care for a puppy or a dog 🐶.")

    async def complete(self, content: str, input_message, response) -> (
            StreamingResponse | BatchResponse):

        await response.send()
        loop = asyncio.get_event_loop()

        # Use loop.run_in_executor to run synchronous method in a separate thread
        history = get_history(content)
        guard_resp = await loop.run_in_executor(None, self._guard, history)
        if guard_resp is not None:
            # short circuit
            return BatchResponse(response=guard_resp)

        # mlflow based chat doesnt support streaming so we need to feed to databricks-genai-sdk
        # cannot do this in async due to compressor
        processed_context = await loop.run_in_executor(None, relevant_question_chain_prompt.invoke,
                                                       history)
        processed_prompt = processed_context["prompt"]
        msgs = [{"content": msg.content, "role": "user"} for msg in processed_prompt.to_messages()]

        buff = []
        token_stream = await loop.run_in_executor(None, run_chat_completion, msgs)
        async for token_chunk in token_stream:
            chunk: HasMessage = token_chunk  # noqa token_chunk is a ChatCompletionChunkObject not Future
            buff.append(chunk.message)
            await response.stream_token(chunk.message)

        sources = processed_context["sources"]
        sources_text = ""
        if len(sources) > 0:
            sources_text += "\n\nSources/Sample Products from PetSmart: \n\n* " + "\n* ".join(sources)
        await response.stream_token(sources_text)

        result = "".join(buff)
        return StreamingResponse(response=result)

    def complete_sync(self, content: str, input_message: cl.Message, response: cl.Message) -> (
            StreamingResponse | BatchResponse):
        try:
            history = get_history(content)
            guard_resp = self._guard(history)
            if guard_resp is not None:
                return BatchResponse(response=guard_resp)
            raise NotImplementedError("Not implemented without streaming")
        except Exception as e:
            return BatchResponse(response=f"Unable to compute response; Error: {str(e)}")
