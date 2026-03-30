# 📄 Smart PDF RAG ChatBot 🤖

This is a simple **Streamlit ChatBot** app powered by **OpenAI**, **LangChain**, **RAG (Retrieval-Augmented Generation)**, and **FAISS** for vector storage.  
The app allows users to **upload a text-based PDF document** and ask natural language questions related to the content of the uploaded file.

---
See Live Demo at : http://217.154.38.177:8501/ 
---

## 🚀 Features

- 🔥 Ask questions from any uploaded PDF document  
- 💡 Uses OpenAI (`gpt-3.5-turbo`) to generate smart answers  
- 🔎 Powered by LangChain's RAG framework and FAISS vector store  
- 📚 Dynamically parses PDFs and generates embeddings  
- 🧠 Retrieval-based contextual answering with document chunking

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) - UI frontend  
- [LangChain](https://www.langchain.com/) - Chain and retrieval logic  
- [OpenAI](https://platform.openai.com/docs/models) - LLM for answer generation  
- [FAISS](https://github.com/facebookresearch/faiss) - Local in-memory vector storage  
- PDF document parsing via `PyPDFLoader`

---

## 📦 Folder Structure

```
pdf_rag_chatbot/
├── app.py             # Main Streamlit app file
└── README.md           # Project documentation
└── Dockerfile          # Dockerfile
└── requirements.txt    # Requirements file
└── keyfile.txt         # OpenAI key stored in a text file (No available on repo, create when running on local) [REQUIRED]
```

---

## 💡 How It Works

1. **Upload** a PDF document (text-based only).  
2. It gets split into chunks using LangChain’s `RecursiveCharacterTextSplitter`.  
3. Each chunk is converted into a vector using OpenAI Embeddings (`text-embedding-3-small` or similar).  
4. Vectors are stored in a temporary FAISS index.  
5. At query time, most relevant chunks are retrieved and passed as context to GPT.  
6. GPT returns an answer grounded in the uploaded document.

---

## ▶️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/siddharthsingh5010/pdf_rag_chatbot
cd pdf_rag_chatbot
```

### 2. Create keyfile.txt

Create a text file in currect directory which stores your OPENAI key


### 3. Builed Docker Image

```bash
docker build -t pdf_rag_app .
```

### 4. Run the app

```bash
docker run -p 8501:8501 pdf_rag_app
```

---

---

## 🔐 Notes

- This app only supports **text-based PDFs** (not scanned images).  
- For best performance, make sure your OpenAI API key has access to `gpt-3.5-turbo`.

---

## 📄 License

MIT License

Author
Abhishek Jain
