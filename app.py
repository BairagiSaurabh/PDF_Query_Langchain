## Import Libraries

import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
#load_dotenv()

api_key = os.getenv("OpenAI_API_KEY")

## Reading the PDF

st.header("Chat with your PDF 💬")
 
pdf = st.file_uploader("Upload your PDF", type='pdf') # upload a PDF file
if pdf is not None:
    pdf_reader = PdfReader(pdf) # read the pdf file
    
    text = "" # collect all text data in this variable
    for page in pdf_reader.pages:
        text += page.extract_text()

    #st.write(text)
        
 ## Forming chunks of data       
        
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # 1000 tokens in each chunk
            chunk_overlap=200, # 2oo tokens will have overlap in consecutive chunks
            length_function=len
            )
    
    chunks = text_splitter.split_text(text=text) # forming and collecting chunks here
    # st.write(chunks)

## Create Embeddings of each chunk of data and store them in the Vector DB
    
    store_name = pdf.name[:-4] # extract the pdf name
    embeddings = OpenAIEmbeddings(openai_api_key = api_key) # using OpenAI to create embeddings

    if os.path.exists(f"{store_name}"): # if already the vector db is present then load it
        #path = f"{store_name}\index.pkl"
        VectorStore = FAISS.load_local(f"{store_name}",embeddings,allow_dangerous_deserialization=True)

        st.write('Vector Database already exists.')

    else:
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings) # providing the input chunks to create embeddings

        VectorStore.save_local(f"{store_name}")
        st.write('Creating new embeddings.')

## Accepting query from user
        
    query = st.text_input("Ask questions about your PDF file:")
    #st.write(query)

    if query:
        docs = VectorStore.similarity_search(query=query, k=3)

        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.success(response)
