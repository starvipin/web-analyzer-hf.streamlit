import streamlit as st


def initialize_session_state():
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_urls" not in st.session_state:
        st.session_state.processed_urls = set()
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None


def parse_urls(urls_input):
    urls = [url.strip() for url in urls_input.strip().split("\n") if url.strip()]
    valid_urls = [
        url for url in urls if url.startswith("http://") or url.startswith("https://")
    ]
    invalid_urls = [url for url in urls if url not in valid_urls]
    return valid_urls, invalid_urls


def render_sidebar(
    openai_api_key,
    load_and_index_urls,
    setup_qa_chain,
    llm_model,
    embedding_model,
):
    with st.sidebar:
        st.header("📚 Add Context URLs")
        urls_input = st.text_area(
            "Enter URLs (one per line):", height=150, key="url_input"
        )
        ingest_button = st.button("Process URLs", key="ingest_button")

        if ingest_button and urls_input:
            valid_urls, invalid_urls = parse_urls(urls_input)

            if invalid_urls:
                st.warning(f"Skipping invalid URLs: {', '.join(invalid_urls)}")

            if not valid_urls:
                st.warning("No valid URLs provided.")
            else:
                new_urls = [
                    url
                    for url in valid_urls
                    if url not in st.session_state.processed_urls
                ]

                if not new_urls:
                    st.warning("All valid URLs entered have already been processed.")
                else:
                    vectorstore = load_and_index_urls(new_urls, openai_api_key)
                    if vectorstore:
                        st.session_state.vector_store = vectorstore
                        st.session_state.processed_urls.update(new_urls)
                        st.session_state.qa_chain = setup_qa_chain(
                            st.session_state.vector_store, openai_api_key
                        )
                    else:
                        st.error("Failed to process the new URLs.")

        st.markdown("---")
        st.markdown(f"**LLM:** `{llm_model}`")
        st.markdown(f"**Embeddings:** `{embedding_model}`")

        if st.session_state.vector_store and openai_api_key:
            status_icon = "Right"
            status_text = f"Ready: {len(st.session_state.processed_urls)} URL(s) processed."
        elif not openai_api_key:
            status_icon = "Key"
            status_text = "API Key needed."
        else:
            status_icon = "!"
            status_text = "Process URLs first."
        st.info(f"{status_icon} {status_text}")


def render_question_area(openai_api_key, setup_qa_chain, logger):
    st.header("❓ Ask a Question")

    if st.session_state.vector_store and openai_api_key:
        if not st.session_state.qa_chain:
            st.session_state.qa_chain = setup_qa_chain(
                st.session_state.vector_store, openai_api_key
            )

        if st.session_state.qa_chain:
            query = st.text_input(
                "Enter your question based on the ingested content:",
                key="query_input",
                placeholder="Ask about the content of the URLs...",
            )

            if query:
                with st.spinner("Asking OpenAI..."):
                    try:
                        result = st.session_state.qa_chain.invoke({"input": query})
                        final_answer = result.get("answer", "").strip()

                        st.subheader("🤖 Answer:")

                        if not final_answer:
                            st.warning("The model returned an empty answer.")
                        elif "i don't know" in final_answer.lower():
                            st.warning(final_answer)
                        else:
                            st.write(final_answer)

                        if "context" in result and result["context"]:
                            with st.expander("📚 Show Sources Used"):
                                for i, doc in enumerate(result["context"]):
                                    source = doc.metadata.get("source", "N/A")
                                    st.info(f"**Source {i + 1}:** `{source}`")
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


def render_processed_urls():
    if st.session_state.processed_urls:
        st.markdown("---")
        with st.expander("Processed URLs in this session:", expanded=False):
            for url in sorted(list(st.session_state.processed_urls)):
                st.markdown(f"- `{url}`")


def render_app(
    openai_api_key,
    load_and_index_urls,
    setup_qa_chain,
    llm_model,
    embedding_model,
    logger,
):
    st.set_page_config(page_title="Web Content Q&A (OpenAI)", layout="wide")
    st.title("💬 Web Content Q&A Tool (OpenAI Powered)")
    st.caption("Ask questions based *only* on the content of the provided webpages.")

    if not openai_api_key:
        st.warning(
            "OpenAI API Key not found. "
            "Please ensure OPENAI_API_KEY is added to your .env file.",
            icon="🔑",
        )

    initialize_session_state()
    render_sidebar(
        openai_api_key,
        load_and_index_urls,
        setup_qa_chain,
        llm_model,
        embedding_model,
    )
    render_question_area(openai_api_key, setup_qa_chain, logger)
    render_processed_urls()
