# 📄 Local PDF RAG Analyst

AI-powered PDF chatbot that enables users to interact with documents using natural language. Built with LangChain, ChromaDB, Ollama, and Streamlit, the system retrieves relevant context from PDF files and generates accurate, context-aware responses using a local LLM.

---

## ✨ Features

- 📑 PDF document ingestion and text chunking
- 🧠 Semantic embeddings using BAAI/bge-small-en-v1.5
- 🔎 Vector similarity search with ChromaDB
- 🤖 Local Llama 3 inference via Ollama
- 💬 Conversation memory (short and long-term context)
- 📚 Source chunk and page tracking
- ⚡ Response latency metrics
- 🖥️ Interactive Streamlit interface
- 🔒 Fully local execution

---

## 🛠 Tech Stack

- Python
- LangChain
- Streamlit
- Ollama
- Llama 3
- ChromaDB
- HuggingFace Embeddings
- PyPDFLoader

---

## ⚙️ Architecture

```text
PDF Document
     ↓
Text Chunking
     ↓
Embeddings
     ↓
ChromaDB Vector Store
     ↓
Retriever
     ↓
Llama 3 (Ollama)
     ↓
Context-Aware Response
     ↓
Source Chunks + Metrics
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/lAlsancakl/local-pdf-rag-analyst.git
cd local-pdf-rag-analyst
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start Ollama

```bash
ollama run llama3
```

### Run the application

```bash
streamlit run app.py
```

---

## 📷 Image

<img width="1000" alt="demo" src="image.png">

---

## 🎯 Key Concepts

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Databases
- Embeddings
- Prompt Engineering
- Local LLM Inference
- Context-Aware Response Generation

---

## 📌 Project Highlights

- Built an end-to-end RAG pipeline for PDF documents.
- Implemented semantic retrieval using embeddings and vector search.
- Integrated a local LLM (Llama 3) for context-grounded responses.
- Added conversation memory, source tracking, and response metrics.
- Designed an interactive Streamlit interface for natural language document interaction.
