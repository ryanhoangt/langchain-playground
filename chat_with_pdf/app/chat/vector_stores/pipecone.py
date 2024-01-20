import os
import pipecone
from langchain.vectorstores import Pinecone
from app.chat.embeddings.openai import embeddings
from app.chat.models import ChatArgs

pipecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PIPECONE_ENV_NAME ')
)

vector_store = Pinecone.from_existing_index(
    os.getenv('PINECONE_INDEX_NAME'), embeddings
)

def build_retriever(chat_args: ChatArgs):
    search_kwargs = {
        "filters": {
            "pdf_id": chat_args.pdf_id
        }
    }

    return vector_store.as_retriever(search_kwargs=search_kwargs)