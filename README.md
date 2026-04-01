# 🌐 Web Content Q&A Tool (Powered by OpenAI & LangChain)

This is a Streamlit-based Web Application that allows users to input webpage URLs, extract their content, and ask questions based **strictly** on the ingested text. It uses a Retrieval-Augmented Generation (RAG) pipeline powered by modern LangChain and OpenAI models.

## ✨ Features

- **Multi-URL Ingestion**: Process single or multiple URLs simultaneously.
- **Smart Text Chunking**: Uses `RecursiveCharacterTextSplitter` for optimal context window management.
- **Vector Database**: Utilizes `FAISS` for fast and efficient similarity search.
- **OpenAI Integration**: Powered by `gpt-4o-mini` for fast responses and `text-embedding-3-small` for embeddings.
- **Hallucination Prevention**: The prompt is strictly engineered to answer *only* from the provided context. If the answer is not in the text, it replies with "I don't know".
- **Source Tracking**: Displays the exact source URLs used to generate the answer.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM Framework**: [LangChain](https://python.langchain.com/) & `langchain-classic`
- **Models**: OpenAI API
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Web Scraping**: BeautifulSoup4 / WebBaseLoader
- **Package Manager**: `uv` (Ultra-fast Python package installer)

## 🚀 Installation & Setup

### Prerequisites
Make sure you have Python installed. We use `uv` for fast dependency management. If you don't have `uv`, install it first:
```bash
pip install uv