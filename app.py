import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- 1. API Configuration (Only for the Chat part) ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# --- 2. Model Initialization ---
# LLM (The brain that answers) - Still uses Gemini Free Tier
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# EMBEDDINGS (The eyes that read the PDF) - HuggingFace (FREE, NO KEY, NO LIMITS)
@st.cache_resource # Cache this so it doesn't reload every time
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- 3. UI Setup ---
st.title("Smart ChatBot: PDF RAG (HuggingFace + Gemini) 🤖")

uploaded_file = st.file_uploader("Upload a PDF to start", type=["pdf"])

if uploaded_file is not None:
    # Save file temporarily
    temp_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- 4. RAG Logic ---
    with st.spinner("Reading and indexing PDF... (This stays free!)"):
        loader = PyPDFLoader(temp_path)
        document = loader.load()
        documents = text_splitter.split_documents(document)

        # No batching needed for HuggingFace! It runs on the server.
        vector = FAISS.from_documents(documents, embeddings)
        retriever = vector.as_retriever()

        prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the context below. 
        If not found, say you don't know.
        
        <context>{context}</context>
        Question: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # --- 5. Chat Interface ---
    user_input = st.text_input("Ask something about the PDF:")

    if user_input:
        with st.spinner("Gemini is generating answer..."):
            try:
                response = retrieval_chain.invoke({"input": user_input})
                st.markdown("### AI Response:")
                st.write(response['answer'])
            except Exception as e:
                st.error(f"Error calling Gemini: {e}")
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
else:
    st.info("Awaiting PDF upload...")
