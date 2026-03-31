import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from langchain_core.callbacks import StdOutCallbackHandler

# --- 1. API Configuration ---
# Retrieve the token from Streamlit Secrets
if "HF_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_TOKEN"]
else:
    st.error("🚨 Missing Hugging Face Token. Please add 'HF_TOKEN' to your Streamlit Secrets.")
    st.stop()

# --- 2. Model Initialization ---
@st.cache_resource
def load_llm():
    # repo_id: The ID of the model on Hugging Face Hub
    # "HuggingFaceH4/zephyr-7b-beta" is a great free chat model
    # "mistralai/Mistral-7B-Instruct-v0.2" is another good option
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    return ChatHuggingFace(llm=llm)

@st.cache_resource
def load_embeddings():
    # This runs locally (CPU) - 100% Free, No Rate Limits
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("📄 PDF ChatBot (Fully Free - Hugging Face) 🤖")
st.markdown("Powered by **Zephyr-7B** (LLM) and **all-MiniLM** (Embeddings)")

# Initialize Models
try:
    llm = load_llm()
    embeddings = load_embeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- 3. File Upload & Processing ---
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    temp_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🧠 Reading PDF and building memory..."):
        try:
            loader = PyPDFLoader(temp_path)
            document = loader.load()
            documents = text_splitter.split_documents(document)

            # Create Vector Store locally
            vector = FAISS.from_documents(documents, embeddings)
            retriever = vector.as_retriever()

            # Define Prompt
            prompt = ChatPromptTemplate.from_template("""
            Answer the user's question based strictly on the context below.
            If the answer is not in the context, reply: "I cannot find this answer in the document."
            
            <context>
            {context}
            </context>

            Question: {input}
            """)

            # Create Chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            st.success("✅ PDF Processed! Ask your question below.")
            
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")

    # --- 4. Chat Interface ---
    user_input = st.text_input("Ask a question about the PDF:")

    if user_input:
        with st.spinner("🤖 Thinking..."):
            try:
                response = retrieval_chain.invoke({"input": user_input})
                st.markdown("### Answer:")
                st.write(response['answer'])
            except Exception as e:
                st.error(f"Error generating response: {e}")
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    st.info("👆 Upload a PDF to start.")
