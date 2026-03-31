import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. API Configuration ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("🚨 Missing Groq API Key. Add 'GROQ_API_KEY' to Streamlit Secrets.")
    st.stop()

# --- 2. Model Initialization ---
@st.cache_resource
def load_llm():
    try:
        # Swapped to the active, upgraded Llama 3.1 model
        return ChatGroq(
            model="llama-3.1-8b-instant", 
            temperature=0
        )
    except Exception as e:
        st.error(f"Failed to load Groq model. Error: {e}")
        st.stop()

@st.cache_resource
def load_embeddings():
    # Runs locally (CPU) - Always free
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("📄 PDF ChatBot (Powered by Groq ⚡)")

# Load models
try:
    llm = load_llm()
    embeddings = load_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- 3. File Upload & State Management ---
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    # Check if we have already processed THIS specific file
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("🧠 Indexing PDF... This only happens once per file!"):
            try:
                # Use tempfile to cleanly handle saving/deleting cross-platform
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                # Load and Process
                loader = PyPDFLoader(temp_path)
                docs = loader.load()
                chunks = text_splitter.split_documents(docs)
                vector = FAISS.from_documents(chunks, embeddings)
                
                # Setup Prompt and Chains
                prompt = ChatPromptTemplate.from_template("""
                Answer the question based strictly on the provided context. 
                If the answer is not found in the context, say "I don't know".
                
                Context: {context}
                Question: {input}
                """)

                document_chain = create_stuff_documents_chain(llm, prompt)
                retrieval_chain = create_retrieval_chain(vector.as_retriever(), document_chain)
                
                # Save to session state so it persists across reruns
                st.session_state.retrieval_chain = retrieval_chain
                st.session_state.current_file = uploaded_file.name
                
                # Clean up the temp file
                os.remove(temp_path)

            except Exception as e:
                st.error(f"Processing Error: {e}")

    # If the chain is ready in session state, allow querying
    if "retrieval_chain" in st.session_state:
        st.success("Ready! Ask questions below.")
        
        user_input = st.chat_input("Ask about the PDF:")
        
        if user_input:
            st.chat_message("user").write(user_input)
            
            try:
                with st.spinner("Thinking..."):
                    res = st.session_state.retrieval_chain.invoke({"input": user_input})
                    st.chat_message("assistant").write(res['answer'])
            except Exception as e:
                st.error(f"Generation Error: {e}")
