import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq

import faiss
import tempfile
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv

torch.classes.__path__ = []

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv(override=True)

# Configurações do Streamlit
st.set_page_config(
    page_title="Converse com documentos",
    page_icon=":books:"
)
st.title("Converse com documentos 📚")

model_class = "groq" # @param ["hf_hub", "openai", "ollama", "groq"]

## Provedores de modelos
def model_hf_hub(model="microsoft/Phi-3-mini-4k-instruct", temperature=0.1):
  llm = HuggingFaceEndpoint(repo_id = model,
      temperature = temperature,
      return_full_text = False,
      max_new_tokens = 1024,
      task="text-generation"
  )
  return llm

def model_openai(model="gpt-4o-mini", temperature=0.1):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
        # demais parâmetros que desejar
    )
    return llm

def model_ollama(model="phi3", temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm

def model_groq(model="llama3-70b-8192", temperature=0.1):
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    return llm

# Indexação e recuperação
def config_retriever(uploads):
    #Carregar documentos
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Divisão em pedaços de texto / split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Armazenamento vetorial
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    # Configuração do retriever
    retriever = vectorstore.as_retriever(seach_type="mmr", 
                                         search_kwargs={"k": 3, 'fetch_k': 4})

    return retriever