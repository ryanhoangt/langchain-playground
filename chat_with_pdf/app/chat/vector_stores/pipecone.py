import os
import pipecone
from langchain.vectorstores import Pinecone
from app.chat.embeddings.openai import embeddings

pipecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PIPECONE_ENV_NAME ')
)

vector_store = Pinecone.from_existing_index(
    os.getenv('PINECONE_INDEX_NAME'), embeddings
)