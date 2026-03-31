import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# --- 1. API Configuration ---
# Make sure to add "GOOGLE_API_KEY" in your Streamlit Secrets!
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# --- 2. Model Initialization ---
# Gemini-1.5-Flash is fast and free
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    task_type="retrieval_document",
    google_api_key=st.secrets["GOOGLE_API_KEY"] # Pass explicitly to avoid env errors
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# --- 3. UI Setup ---
st.title("Smart ChatBot: Ask Questions from PDF Documents 🤖")
st.markdown("[💻 View Source on GitHub](https://github.com)")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save file temporarily to read it
    temp_file_path = os.path.join(os.getcwd(), uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # --- 4. RAG Logic ---
    with st.spinner("Processing PDF..."):
        # Load and Split Document
        loader = PyPDFLoader(temp_file_path)
        document = loader.load()
        documents = text_splitter.split_documents(document)

        # FIX: Batch Processing to avoid "GoogleGenerativeAIError"
        # Google API Limit: 100 chunks per request. We use 20 to be safe.
        batch_size = 20
        vector = None
        
        import time
        
        # Create a progress bar
        progress_text = "Embedding document chunks..."
        my_bar = st.progress(0, text=progress_text)
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            
            # First batch creates the vector store, subsequent batches add to it
            if vector is None:
                vector = FAISS.from_documents(batch, embeddings)
            else:
                vector.add_documents(batch)
            
            # Update progress bar
            progress = min((i + batch_size) / len(documents), 1.0)
            my_bar.progress(progress, text=f"Embedded {i + len(batch)}/{len(documents)} chunks")
            
            # CRITICAL: Sleep to respect Free Tier rate limits
            time.sleep(2)

        # Clear progress bar
        my_bar.empty()

        # Setup Retriever
        retriever = vector.as_retriever()

        # Define the Prompt
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        If the answer is not in the context, say "I cannot find the answer in the document."
        
        <context>
        {context}
        </context>

        Question: {input}
        """)

        # Create Chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)


    # --- 5. Chat Interface ---
    st.subheader("Ask a Question")
    user_input = st.text_input("Query:", placeholder="e.g., What is the main topic of this document?")

    if user_input:
        with st.spinner("Gemini is thinking..."):
            response = retrieval_chain.invoke({"input": user_input})
            st.markdown("### AI Response:")
            st.write(response['answer'])
    
    # Cleanup: remove the temp file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

else:
    st.info("Please upload a PDF file to start chatting.")
