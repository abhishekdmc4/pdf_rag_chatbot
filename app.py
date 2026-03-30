### Description ###
# This is a simple Streamlit ChatBot app which is powered by OpenAI Model along with LangChain, RAG
# In this user can upload any text PDF file and can ask chatbot questions regarding content of PDF in natural language
####

# Importing Libraries
import streamlit as st
import os
import time
## from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import openai
from langchain.agents import load_tools,initialize_agent,AgentType,create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Getting and setting Open AI API Key - This is required to use OpenAI model
with open("keyfile.txt") as f:
    key = f.read().strip()
os.environ["OPENAI_API_KEY"] = key

# Defining LLM Model and Splitter
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
text_splitter = RecursiveCharacterTextSplitter()

# Title of the app
st.title("Smart ChatBot : Ask Questions from PDF Documents🤖")

st.markdown(
    "[💻 View Source on GitHub](https://github.com/siddharthsingh5010/pdf_rag_chatbot)", 
    unsafe_allow_html=True
)
st.markdown(
    "[Owner : Siddharth Singh](https://www.nomadicsid.com)", 
    unsafe_allow_html=True
)
# File upload widget
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

# Display information about the uploaded file
if uploaded_file is not None:
    # To read file as bytes
    file_bytes = uploaded_file.read()
    st.write(f"Uploaded file: {uploaded_file.name}")
    st.write(f"File type: {uploaded_file.type}")
    # Saving file to disk
    save_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show chat box after file upload
    st.subheader("LLM and RAG Powered ChatBot")
    user_input = st.text_input("You: ", "Type your message here...")

    # document loader
    loader = PyPDFLoader(f"{uploaded_file.name}")
    document = loader.load()
    documents = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)   # document chain being part of the retrieval Chain
    if user_input!='Type your message here...':
      response = retrieval_chain.invoke({"context": "You are an AI assistant who need to examine content of document and provide concise answers",
                                   "input": user_input})
      st.text_area("AI : ", value=response['answer'], height=200)
else:
    st.write("No file uploaded yet.")
