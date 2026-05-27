import os
import streamlit as st
import logging
from dotenv import load_dotenv

# --- Load .env variables ---
# Ye line aapki .env file se OPENAI_API_KEY utha legi
load_dotenv()

# --- Modern LangChain Imports for OpenAI ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from ui import render_app

# --- ENV FIXES ---
os.environ["STREAMLIT_WATCHER_TYPE"] = "watchdog"
os.environ["USER_AGENT"] = "MyWebAnalyzerApp/1.0"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
LLM_MODEL = "gpt-4o-mini" # Fast and cost-effective model
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Helper Functions ---
@st.cache_resource(show_spinner="Loading and processing URLs...")
def load_and_index_urls(urls, api_key):
    """Loads URLs, splits text, creates embeddings, and indexes in FAISS."""
    if not api_key:
        st.error("OpenAI API Key is required for embeddings.")
        return None
        
    try:
        loader = WebBaseLoader(urls, requests_per_second=2, continue_on_failure=True)
        loader.requests_kwargs = {'timeout': 20}
        docs = loader.load()

        if not docs:
            st.error("Could not load any content from the provided URLs. Check URLs, permissions, and connection.")
            return None

        docs = [doc for doc in docs if len(doc.page_content) > 50]
        if not docs:
            st.warning("Content loaded, but it might be too short after filtering.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        if not splits:
            st.error("Could not split the documents into chunks.")
            return None

        logger.info(f"Embedding {len(splits)} text chunks using {EMBEDDING_MODEL}...")
        
        # Initialize OpenAI Embeddings
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        st.success(f"Successfully processed {len(docs)} URL(s) into {len(splits)} text chunks.")
        return vectorstore

    except Exception as e:
        st.error(f"An error occurred during ingestion: {e}")
        logger.error(f"Ingestion error: {e}", exc_info=True)
        return None

def setup_qa_chain(_vector_store, api_key):
    """Sets up the LangChain LCEL retrieval chain using OpenAI."""
    if not api_key:
        st.error("OpenAI API Key not found.")
        return None
    try:
        logger.info(f"Setting up QA chain with model: {LLM_MODEL}")

        # Modern ChatPromptTemplate approach
        system_prompt = (
            "You are an assistant designed to answer questions based ONLY on the provided context.\n"
            "Carefully review the following context.\n"
            "If the answer to the question is present in the context, provide that answer cleanly.\n"
            "If the answer is not found in the context, you MUST respond with 'I don't know'.\n"
            "Do not add any information that is not explicitly stated in the context.\n\n"
            "CONTEXT:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1, 
            openai_api_key=api_key,
            max_tokens=300
        )

        retriever = _vector_store.as_retriever(search_kwargs={"k": 4})

        # Create LCEL Chain
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        logger.info("QA chain setup complete.")
        return qa_chain

    except Exception as e:
        st.error(f"Failed to setup QA chain: {e}")
        logger.error(f"QA Chain Setup Error: {e}", exc_info=True)
        return None

# --- Get API Token ---
# The load_dotenv() at the top will put the key into os.getenv
openai_api_key = os.getenv("OPENAI_API_KEY")

render_app(
    openai_api_key=openai_api_key,
    load_and_index_urls=load_and_index_urls,
    setup_qa_chain=setup_qa_chain,
    llm_model=LLM_MODEL,
    embedding_model=EMBEDDING_MODEL,
    logger=logger,
)
