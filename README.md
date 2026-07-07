# ⚖️ LexAI — Legal Rights Assistant for Indian Citizens

An AI-powered legal rights assistant that helps common Indian citizens understand their legal rights without needing a lawyer. Built with RAG (Retrieval-Augmented Generation) + Machine Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange)
![RAG](https://img.shields.io/badge/RAG-ChromaDB-purple)

---

## 🎯 Problem Statement

Most Indian citizens don't know their basic legal rights. A lawyer consultation costs ₹2,000–₹10,000 per hour — unaffordable for many. LexAI solves this by providing instant, free, AI-powered legal guidance in plain language.

---

## 🚀 Features

- 🔍 **Natural language queries** — describe your problem in plain English
- 🧠 **ML Domain Classifier** — automatically identifies legal domain
- 📚 **RAG Pipeline** — retrieves exact relevant sections from Indian law
- ⚡ **AI-powered answers** — generates simple, actionable explanations
- 💬 **Chat-style interface** — modern, professional UI
- 📋 **Copy and Print** — save your legal assessment
- 🕐 **Query history** — tracks your recent queries
- 📱 **Mobile responsive** — works on all devices

---

## 🏗️ System Architecture

User Query → ML Classifier → RAG Pipeline → LLM → Flask Web App

- ML Classifier identifies legal domain using Random Forest + TF-IDF
- RAG Pipeline retrieves relevant law sections from ChromaDB
- LLM generates clear explanation using Groq LLaMA 3.3
- Flask Web App displays result with source citations

---

## 📚 Law Documents Covered

| Law | Domain |
|-----|--------|
| Indian Penal Code, 1860 | Criminal Law |
| Industrial Disputes Act, 1947 | Labour Law |
| Income Tax Act, 1961 | Tax Law |
| Code of Civil Procedure, 1908 | Civil Law |
| Transfer of Property Act, 1882 | Property Law |
| Consumer Protection Act, 2019 | Consumer Law |
| Constitution of India | Constitutional Law |
| Right to Information Act, 2005 | RTI Law |

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| ML Classifier | Scikit-learn, Random Forest, TF-IDF |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| LLM | Groq API (LLaMA 3.3 70B) |
| Web Framework | Flask |
| PDF Processing | PyPDFLoader, pdfplumber |

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 62.17% |
| Random Forest | 69.26% (Best) |
| SVM | 65.62% |

- Training data: 10,000 balanced samples across 5 legal domains
- Vector store: 14,391 chunks from 8 official Indian law documents

---

## ⚙️ Installation and Setup

### 1. Clone the repository
git clone https://github.com/spider061/legal-rights-assistant.git
cd legal-rights-assistant

### 2. Install dependencies
pip install langchain langchain-community chromadb sentence-transformers
pip install flask scikit-learn pandas numpy pypdf pdfplumber
pip install tqdm datasets groq python-dotenv

### 3. Set up environment variables
Create a .env file in the root directory:
GROQ_API_KEY=your-groq-api-key-here

Get your free Groq API key at: https://console.groq.com

### 4. Prepare the data
python prepare_data.py
python extract_pdf_data.py

### 5. Train the ML model
python train_model.py

### 6. Build the RAG pipeline
python rag_pipeline.py

### 7. Run the web app
python app.py

Open http://localhost:5000 in your browser.

---

## 📁 Project Structure

legal-rights-assistant/
├── data/
│   ├── raw/               Training datasets
│   └── laws/              Indian law PDFs
├── models/                Saved ML model
├── vectorstore/           ChromaDB vector database
├── templates/             Flask HTML templates
├── app.py                 Flask web application
├── train_model.py         ML training script
├── rag_pipeline.py        RAG pipeline builder
├── query_engine.py        Query processing engine
├── prepare_data.py        Data preparation script
└── extract_pdf_data.py    PDF text extraction

---

## 🌟 Example Queries

- My employer has not paid my salary for 3 months
- My landlord is not returning my security deposit
- I bought a defective product online, what are my rights?
- I was wrongfully terminated without notice
- What are my rights if I am arrested by police?
- How do I file an RTI application?

---

## ⚠️ Disclaimer

This application provides general legal information based on publicly available Indian law documents. It is not a substitute for professional legal advice. For serious legal matters, please consult a qualified lawyer.

---

## 👨‍💻 Developer

Zarik Rasool
B.Tech AI/ML — Semester IV
Model Institute of Engineering and Technology (MIET), Jammu

---

## 📄 License

This project is open source and available under the MIT License.