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

    #The question is the last entry of the history
    def extract_question(self, input):
        return input[-1]["content"]

    # Truncate history
    def extract_history(self, input):
        return self.truncate_chat_history(input[:-1])
    
    def format_context(self, docs):
        return "\n\n".join([d.page_content for d in docs])

    def extract_source_urls(self, docs):
        return list(set([d.metadata["url"] for d in docs]))
    
    def extract_source_titles(self, docs):
        return list(set([d.metadata["title"] for d in docs]))

# class RetrieveData:
#     def __init__(self, host, token, endpoint_name, index_name, embedding_model, text_column, columns, k=3):
#         """
#         This function creates and returns a retriever object for a given vector search index specified by the input parameters.
#         Args:
#         - host: A string, the workspace URL
#         - token: A string, the personal access token
#         - endpoint_name: A string, the name of the vector search endpoint that has been created
#         - index_name: A string, name of the index being queried
#         - embedding_model: An instance of an embedding model to be used to encode the textual data
#         - text_column: A string, the name of the field in the index that contains the text data
#         - columns: A list, names of columns to retrieve from the index
#         - k: An int, number of records to retrieve from the index.
#         Returns:
#         - A retriever object for the specified index.
#         """
#         self.host = host
#         self.token = token
#         self.endpoint_name = endpoint_name
#         self.index_name = index_name
#         self.embedding_model = embedding_model
#         self.text_column = text_column
#         self.columns = columns
#         self.k = k

#         self.retriever = self.get_retriever()

#     def get_retriever(self, persist_dir: str = None):

#         from databricks.vector_search.client import VectorSearchClient
#         from langchain.vectorstores import DatabricksVectorSearch

#         # Get the vector search index
#         vsc = VectorSearchClient(workspace_url=self.host, personal_access_token=self.token)
#         vs_index = vsc.get_index(endpoint_name=self.endpoint_name, index_name=self.index_name)

#         # Create the retriever
#         vectorstore = DatabricksVectorSearch(
#             vs_index, text_column=self.text_column, embedding=self.embedding_model, columns=self.columns
#         )
#         return vectorstore.as_retriever(search_kwargs={"k": self.k})