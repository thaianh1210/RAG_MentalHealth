import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import toml 
from llama_index.core.extractors import SummaryExtractor
import openai
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import streamlit as st 
from src.global_settings import STORAGE_PATH, FILES_PATH, CACHE_FILE
from src.prompts import CUSTOM_SUMMARY_EXTRACT_TEMPLATE


import toml

# Đọc API key từ file secrets.toml
secrets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "streamlit", "secrets.toml")
secrets = toml.load(secrets_path)
openai_api_key = secrets["openai"]["OPENAI_API_KEY"]

# Đặt API key cho OpenAI
openai.api_key = openai_api_key
os.environ["OPENAI_API_KEY"] = openai_api_key

# Cấu hình Settings
Settings.llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(api_key=openai_api_key)

def ingest_documents():
    documents = SimpleDirectoryReader(
        input_files = FILES_PATH,
        filename_as_id= True
    ).load_data()
    for doc in documents:
        print(doc.id_)
    try:
        cached_hashes = IngestionCache.from_persist_path(
            CACHE_FILE
        )
        print("Cache file found. Running using cache...")
    except:
        cached_hashes = None
        print("No cache file found. Running without cache...")
    pipeline = IngestionPipeline(
        transformations= [TokenTextSplitter(chunk_size=512, chunk_overlap=20), 
                          SummaryExtractor(summaries=['self'], prompt_template= CUSTOM_SUMMARY_EXTRACT_TEMPLATE), 
                          OpenAIEmbedding(api_key=openai_api_key)],
        cache = cached_hashes
    )
    
    nodes = pipeline.run(documents = documents)
    if cached_hashes:
        pipeline.cache.persist(CACHE_FILE)
    
    return nodes

if __name__ == "__main__":
    ingest_documents()
