import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# These specific imports fix the "ModuleNotFoundError"
# Import from specific submodules
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# --- 1. API Configuration ---
if "HF_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]
else:
    st.error("🚨 Missing Hugging Face Token. Add 'HF_TOKEN' to Streamlit Secrets.")
    st.stop()

# --- 2. Model Initialization ---
@st.cache_resource
def load_llm():
    # "microsoft/Phi-3-mini-4k-instruct" is currently the most reliable FREE model
    # Backup option if this fails: "google/gemma-1.1-2b-it"
    repo_id = "microsoft/Phi-3-mini-4k-instruct"
    
    try:
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        return ChatHuggingFace(llm=llm)
    except Exception as e:
        st.error(f"Failed to load model '{repo_id}'. Error: {e}")
        st.stop()

@st.cache_resource
def load_embeddings():
    # Runs locally (CPU) - Always free, always works
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("📄 PDF ChatBot (Free Hugging Face API) 🤖")

# Load models
try:
    llm = load_llm()
    embeddings = load_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# --- 3. File Upload ---
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    temp_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🧠 Indexing PDF..."):
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            chunks = text_splitter.split_documents(docs)
            vector = FAISS.from_documents(chunks, embeddings)
            
            prompt = ChatPromptTemplate.from_template("""
            Answer based on context. If not found, say "I don't know".
            
            Context: {context}
            Question: {input}
            """)

            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(vector.as_retriever(), document_chain)
            
            st.success("Ready! Ask questions below.")
        except Exception as e:
            st.error(f"Processing Error: {e}")

    user_input = st.text_input("Ask about the PDF:")
    if user_input:
        try:
            res = retrieval_chain.invoke({"input": user_input})
            st.write(res['answer'])
        except Exception as e:
            st.error(f"Generation Error (API might be busy): {e}")
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
