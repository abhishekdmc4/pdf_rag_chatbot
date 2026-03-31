# 📄 Smart PDF RAG ChatBot 🤖 (Powered by Groq & Llama 3.3)

This is a lightning-fast **Streamlit ChatBot** app powered by **Groq**, **Meta's Llama 3**, **LangChain**, **RAG (Retrieval-Augmented Generation)**, and **FAISS** for vector storage.  
The app allows users to **upload a text-based PDF document** and ask natural language questions related to the content of the uploaded file, completely free of charge.

---
See Live Demo at : https://abhishek-pdf-rag-chatbot.streamlit.app/
---

## 🚀 Features

- 🔥 Ask questions from any uploaded PDF document.
- ⚡ Lightning-fast inference powered by **Groq** using Meta's state-of-the-art `llama-3.3-70b-versatile` model.
- 🆓 100% Free, local document embeddings using Hugging Face (`all-MiniLM-L6-v2`) — no API limits!
- 🔎 Powered by the modern LangChain 0.3 RAG framework and FAISS vector store.
- 🧠 Smart session state management to ensure PDFs are only indexed once per upload, saving time and memory.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) - UI frontend and state management
- [LangChain](https://www.langchain.com/) - Orchestration and retrieval logic
- [Groq](https://groq.com/) - High-speed LLM inference engine
- [HuggingFace](https://huggingface.co/) - Local, CPU-friendly embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Local in-memory vector storage
- PDF document parsing via `PyPDFLoader`

---

## 📦 Folder Structure

    pdf_rag_chatbot/
    ├── app.py              # Main Streamlit app file
    ├── README.md           # Project documentation
    ├── Dockerfile          # Dockerfile
    ├── requirements.txt    # Dependencies (CPU-optimized)
    └── .streamlit/         # Directory for local secrets (Create this)
        └── secrets.toml    # Store your GROQ_API_KEY here [REQUIRED]

---

## 💡 How It Works

1. **Upload** a PDF document (text-based only).  
2. The document is split into smaller, readable chunks using LangChain’s `RecursiveCharacterTextSplitter`.  
3. Each chunk is converted into a vector using local Hugging Face Embeddings running right on your CPU.  
4. These vectors are temporarily stored in a FAISS index.  
5. At query time, the most relevant chunks are retrieved and passed as context to the Llama 3.3 model.  
6. Groq processes the context and returns a highly accurate, grounded answer in milliseconds.

---

## ▶️ Getting Started (Local Deployment)

### 1. Clone the repository

```bash
git clone [https://github.com/abhishekdmc4/pdf_rag_chatbot](https://github.com/abhishekdmc4/pdf_rag_chatbot)
cd pdf_rag_chatbot
```

### 2. Set Up Your API Key

Since the app uses Streamlit's native secrets manager, create a `.streamlit` folder in the root directory and add a `secrets.toml` file to store your free Groq API key:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_your_api_key_here..."
```

### 3. Build Docker Image

```bash
docker build -t pdf_rag_app .
```

### 4. Run the app

Run the container, making sure to mount your secrets folder so the app can read your API key:

```bash
docker run -p 8501:8501 -v $(pwd)/.streamlit:/app/.streamlit pdf_rag_app
```

*(Note: The `-v` flag ensures Docker can access the `.streamlit` folder you created in step 2).*

---

## ☁️ Deploying to Streamlit Community Cloud (Free)

If you want to host this on the web instead of running Docker:
1. Push your code to a public or private GitHub repository.
2. Log into [Streamlit Community Cloud](https://streamlit.io/cloud) and create a new app pointing to your `app.py`.
3. Go to the app's **Advanced Settings > Python Version** and ensure it is set to **Python 3.11** or **3.12**.
4. Go to the app's **Settings > Secrets** and paste in your API key: 
   ```toml
   GROQ_API_KEY = "gsk_your_api_key_here..."
   ```

---

## 🔐 Notes

- This app only supports **text-based PDFs** (not scanned images).  
- Because the embeddings run locally, the initial PDF processing might take a few seconds, but all subsequent questions will be answered almost instantly!

---

## 📄 License

MIT License

Author
Abhishek Jain
