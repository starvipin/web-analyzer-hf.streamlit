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

# --- Streamlit UI ---
st.set_page_config(page_title="Web Content Q&A (OpenAI)", layout="wide")
st.title("💬 Web Content Q&A Tool (OpenAI Powered)")
st.caption("Ask questions based *only* on the content of the provided webpages.")

# --- Get API Token ---
# The load_dotenv() at the top will put the key into os.getenv
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.warning(
        "OpenAI API Key not found. "
        "Please ensure OPENAI_API_KEY is added to your .env file.",
        icon="🔑"
    )

# --- Session State Initialization ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = set()
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# --- Sidebar ---
with st.sidebar:
    st.header("📚 Add Context URLs")
    urls_input = st.text_area("Enter URLs (one per line):", height=150, key="url_input")
    ingest_button = st.button("Process URLs", key="ingest_button")

    if ingest_button and urls_input:
        urls = [url.strip() for url in urls_input.strip().split('\n') if url.strip()]
        valid_urls = [url for url in urls if url.startswith('http://') or url.startswith('https://')]
        invalid_urls = [url for url in urls if url not in valid_urls]

        if invalid_urls:
            st.warning(f"Skipping invalid URLs: {', '.join(invalid_urls)}")

        if not valid_urls:
             st.warning("No valid URLs provided.")
        else:
            new_urls = [url for url in valid_urls if url not in st.session_state.processed_urls]

            if not new_urls:
                st.warning("All valid URLs entered have already been processed.")
            else:
                vectorstore = load_and_index_urls(new_urls, openai_api_key)
                if vectorstore:
                    st.session_state.vector_store = vectorstore
                    st.session_state.processed_urls.update(new_urls)
                    st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store, openai_api_key)
                else:
                    st.error("Failed to process the new URLs.")

    st.markdown("---")
    st.markdown(f"**LLM:** `{LLM_MODEL}`")
    st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    
    if st.session_state.vector_store and openai_api_key:
        status_icon = "✅"
        status_text = f"Ready: {len(st.session_state.processed_urls)} URL(s) processed."
    elif not openai_api_key:
        status_icon = "🔑"
        status_text = "API Key needed."
    else:
        status_icon = "⚠️"
        status_text = "Process URLs first."
    st.info(f"{status_icon} {status_text}")

# --- Main Area ---
st.header("❓ Ask a Question")

if st.session_state.vector_store and openai_api_key:
    if not st.session_state.qa_chain:
         st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store, openai_api_key)

    if st.session_state.qa_chain:
        query = st.text_input("Enter your question based on the ingested content:", key="query_input", placeholder="Ask about the content of the URLs...")

        if query:
            with st.spinner("Asking OpenAI..."):
                try:
                    result = st.session_state.qa_chain.invoke({"input": query})
                    
                    # Modern LangChain automatically places the clean response in 'answer'
                    final_answer = result.get('answer', "").strip()

                    st.subheader("🤖 Answer:")

                    if not final_answer:
                         st.warning("The model returned an empty answer.")
                    elif "i don't know" in final_answer.lower():
                         st.warning(final_answer) 
                    else:
                        st.write(final_answer)

                    # Show sources
                    if 'context' in result and result['context']:
                         with st.expander("📚 Show Sources Used"):
                             for i, doc in enumerate(result['context']):
                                 source = doc.metadata.get('source', 'N/A')
                                 st.info(f"**Source {i+1}:** `{source}`")
                                 st.text(doc.page_content[:500] + "...")
                                 st.markdown("---")
                    else:
                         logger.info("No source documents returned.")

                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    logger.error(f"Query Error: {e}", exc_info=True)
    else:
         st.error("QA Chain could not be initialized.")

elif not openai_api_key:
     st.warning("Please provide your OpenAI API Key to enable Q&A.", icon="🔑")
else:
    st.info("Please process some URLs using the sidebar first.", icon="↖️")

# --- Display Processed URLs ---
if st.session_state.processed_urls:
    st.markdown("---")
    with st.expander("Processed URLs in this session:", expanded=False):
        for url in sorted(list(st.session_state.processed_urls)):
            st.markdown(f"- `{url}`")