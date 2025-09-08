import boto3
import json
import os
import sys


## Using Titan Embedding model to create vectors or to generate embedding

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


## Vector Embeddings And Vector store

from langchain_community.vectorstores import FAISS


## LLM models

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock clients

bedrock = boto3.client(service_name="bedrock-runtime",
        aws_access_key_id="AKIATNRWB2SEVZXJIGUF",
        aws_secret_access_key="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",
        region_name="us-east-1",
)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


## Data ingestion steps

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    ## - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)

    docs = text_splitter.split_documents(documents)
    return docs


## Vextor embeddings and vector stores

def get_vector_store(docs):
    vector_store_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )

    vector_store_faiss.save_local("faiss_index")


def get_mistral_llm():
    ## Creating mistral model
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock,model_kwargs={"max_tokens":512})

    return llm

def get_llama3_llm():
    ## Creating mistral model
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock,model_kwargs={"max_gen_len":512})

    return llm


## Prompt template

prompt_template = """

Human : Use the following pieces of context to provide a concise answer
to the question at the end but use atleast summarize with 250 words
with detailed explanations. If you don't know the answer, just say that 
you don't know, don't try to make up the answer


<context>
{context}
</context

Question : {question}

Assistant:
"""

Prompt = PromptTemplate(
    template=prompt_template,input_variables=["context","question"]

)

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type ="stuff",
        retriever = vectorstore_faiss.as_retriever(
            search_type = "similarity" , search_kwargs={"k":3}
        ),

        return_source_documents = True,
        chain_type_kwargs = {"prompt":Prompt}
    )

    answer = qa({"query":query})
    return answer["result"]



## Streamlit app

import streamlit as st

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")

    user_question = st.text_input("Ask a Question from PDF files")


    with st.sidebar:
        st.title("Update or create Vector store")

        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    
    if st.button("Mistral Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
            llm = get_mistral_llm()

            ## Faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

   
    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index",bedrock_embeddings,allow_dangerous_deserialization=True)
            llm = get_llama3_llm()

            ## Faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))


if __name__ == "__main__":
    main()



