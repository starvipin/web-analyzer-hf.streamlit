import os
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging

# --- ENV FIXES (Keep these) ---
os.environ["STREAMLIT_WATCHER_TYPE"] = "watchdog"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
LLM_MODEL_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Helper Functions ---
@st.cache_resource(show_spinner="Loading and processing URLs...")
def load_and_index_urls(urls):
    """Loads URLs, splits text, creates embeddings, and indexes in FAISS."""
    try:
        loader = WebBaseLoader(urls,
                               requests_per_second=2,
                               continue_on_failure=True)
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
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(splits, embeddings)
        st.success(f"Successfully processed {len(docs)} URL(s) into {len(splits)} text chunks.")
        return vectorstore

    except Exception as e:
        st.error(f"An error occurred during ingestion: {e}")
        logger.error(f"Ingestion error: {e}", exc_info=True)
        return None

# Updated function to use HuggingFaceHub and a specific prompt
def setup_qa_chain(_vector_store, api_token):
    """Sets up the LangChain RetrievalQA chain using HuggingFaceHub LLM."""
    if not api_token:
        st.error("Hugging Face API Token not found. Please set the HUGGINGFACEHUB_API_TOKEN.")
        return None
    try:
        logger.info(f"Setting up QA chain with model: {LLM_MODEL_REPO_ID} via HuggingFaceHub")

        # Define the prompt template for grounded Q&A
        prompt_template_str = """
You are an assistant designed to answer questions based ONLY on the provided context.
Carefully review the following context.
If the answer to the question is present in the context, provide that answer.
If the answer is not found in the context, you MUST respond with "I don't know".
Do not add any information that is not explicitly stated in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
        PROMPT = PromptTemplate(
            template=prompt_template_str, input_variables=["context", "question"]
        )

        # Instantiate the HuggingFaceHub LLM
        llm = HuggingFaceHub(
            repo_id=LLM_MODEL_REPO_ID,
            huggingfacehub_api_token=api_token,
            task="text-generation", # Correct task for instruct models
            model_kwargs={
                "max_new_tokens": 300,   # Limit response length
                "temperature": 0.1,    # Lower temperature for more factual, less creative response
                "top_p": 0.9,         # Nucleus sampling
                "do_sample": False,    # Turn off sampling for more deterministic, context-focused answers
                "repetition_penalty": 1.1 # Slightly discourage repetition
            }
        )

        retriever = _vector_store.as_retriever(search_kwargs={"k": 4}) # Retrieve 4 chunks

        # Create the RetrievalQA chain with the custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # "stuff" puts all context into the prompt
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT} # Inject the custom prompt
        )
        logger.info("QA chain setup complete with custom prompt.")
        return qa_chain

    except Exception as e:
        st.error(f"Failed to setup QA chain: {e}")
        logger.error(f"QA Chain Setup Error: {e}", exc_info=True)
        return None

# --- Streamlit UI (Mostly unchanged from previous version) ---
st.set_page_config(page_title="Web Content Q&A (HF Hub)", layout="wide")
st.title("💬 Web Content Q&A Tool (using HF Hub)")
st.caption("Ask questions based *only* on the content of the provided webpages. The model will say 'I don't know' if the answer isn't found.")

# --- Get API Token ---
hf_api_token = None
try:
    hf_api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
except (FileNotFoundError, AttributeError):
    logger.info("Streamlit secrets not found, checking environment variables.")
    hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_api_token:
    st.warning(
        "Hugging Face API Token not found. "
        "Please set the `HUGGINGFACEHUB_API_TOKEN` environment variable or add it to Streamlit secrets."
        "\n\nGet a token here: https://huggingface.co/settings/tokens",
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
                vectorstore = load_and_index_urls(new_urls)
                if vectorstore:
                    # Replace or merge logic (replace shown)
                    st.session_state.vector_store = vectorstore
                    st.session_state.processed_urls.update(new_urls)
                    st.success(f"Processed {len(new_urls)} new URL(s). Ready for questions!")
                    # Re-create the QA chain with the updated store and token
                    st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store, hf_api_token)
                else:
                    st.error("Failed to process the new URLs.")

    st.markdown("---")
    st.markdown(f"**LLM:** `{LLM_MODEL_REPO_ID}` (via HF Hub)")
    st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}` (Local)")
    if st.session_state.vector_store and hf_api_token:
        status_icon = "✅"
        status_text = f"Ready: {len(st.session_state.processed_urls)} URL(s) processed."
    elif not hf_api_token:
        status_icon = "🔑"
        status_text = "API Token needed."
    else:
        status_icon = "⚠️"
        status_text = "Process URLs first."
    st.info(f"{status_icon} {status_text}")


# --- Main Area ---
st.header("❓ Ask a Question")

if st.session_state.vector_store and hf_api_token:
    if not st.session_state.qa_chain:
         # Attempt to setup chain if it wasn't created during URL processing (e.g., if token was missing then)
         st.session_state.qa_chain = setup_qa_chain(st.session_state.vector_store, hf_api_token)

    if st.session_state.qa_chain:
        query = st.text_input("Enter your question based on the ingested content:", key="query_input", placeholder="Ask about the content of the URLs...")

        if query:
            with st.spinner(f"Asking `{LLM_MODEL_REPO_ID.split('/')[1]}`..."): # Show model name only
                try:
                    result = st.session_state.qa_chain.invoke({"query": query})
                    raw_output = result.get('result', "") # Get the full raw output

                    # --- ADD PARSING LOGIC HERE ---
                    answer_marker = "ANSWER:"
                    # Find the last occurrence of the marker
                    marker_position = raw_output.rfind(answer_marker)

                    if marker_position != -1:
                        # Extract text *after* the marker
                        final_answer = raw_output[marker_position + len(answer_marker):].strip()
                    else:
                        # Fallback if "ANSWER:" isn't found (shouldn't happen with current prompt)
                        logger.warning("Could not find 'ANSWER:' marker in the LLM output. Displaying raw output.")
                        final_answer = raw_output.strip()
                    # --- END PARSING LOGIC ---

                    st.subheader("🤖 Answer:")

                    # Now check the *parsed* answer
                    if not final_answer:
                         st.warning("The model returned an empty answer after parsing.")
                    elif "i don't know" in final_answer.lower():
                         st.warning(final_answer) # Display as warning if it doesn't know
                    else:
                        st.write(final_answer) # Display the clean, parsed answer

                    if 'source_documents' in result and result['source_documents']:
                         with st.expander("📚 Show Sources Used"):
                             for i, doc in enumerate(result['source_documents']):
                                 source = doc.metadata.get('source', 'N/A')
                                 st.info(f"**Source {i+1}:** `{source}`")
                                 st.text(doc.page_content[:500] + "...")
                                 st.markdown("---")
                    else:
                         # This might happen if the LLM says "I don't know" as per instructions
                         logger.info("No source documents returned, potentially because the answer was 'I don't know'.")
                         # st.warning("No source documents were retrieved or returned.")


                except Exception as e:
                    st.error(f"An error occurred while getting the answer: {e}")
                    logger.error(f"Query Error: {e}", exc_info=True)
    else:
         st.error("QA Chain could not be initialized. Check configuration and API token.")

elif not hf_api_token:
     st.warning("Please provide your Hugging Face API Token to enable Q&A.", icon="🔑")
else:
    st.info("Please process some URLs using the sidebar first.", icon="↖️")

# --- Display Processed URLs ---
if st.session_state.processed_urls:
    st.markdown("---")
    with st.expander("Processed URLs in this session:", expanded=False):
        for url in sorted(list(st.session_state.processed_urls)):
            st.markdown(f"- `{url}`")