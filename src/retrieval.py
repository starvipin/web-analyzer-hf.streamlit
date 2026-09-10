"""Rank page chunks and answer questions using the selected context."""

import logging
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from . import config
from .config import LLM_MODEL, OPENAI_MAX_RETRIES, OPENAI_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are LinkMind AI, developed by Vipin.\n"
    "Answer questions about the user's ingested link using ONLY the provided context.\n"
    "Understand casual, misspelled, or short questions by mapping them to the closest meaning in context.\n"
    "The context or user question may be in Hindi, English, or Hinglish. Answer in the same language style the user uses.\n"
    "If the page is in Hindi and the user asks in Hinglish, translate the meaning internally and answer from the Hindi context.\n"
    "Always use exact model names, product names, versions, and technical details - never summarize them.\n"
    "If the answer isn't in the context, briefly state what topics you can help with from this link "
    "and ask for a more specific question. Do not say 'I don't know'.\n"
    "If asked about this app or its developer, say it was developed by Vipin.\n\n"
    "CONTEXT:\n{context}"
)


def tokenize(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[\w\u0900-\u097F][\w\u0900-\u097F_-]+", text.lower(), flags=re.UNICODE)
        if len(word) > 1
    }


def char_ngrams(text: str, n: int = 3) -> set[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    return {normalized[index:index + n] for index in range(max(len(normalized) - n + 1, 0))}


class LocalRetrievalQA:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.doc_tokens = [tokenize(doc.page_content) for doc in docs]
        self.doc_ngrams = [char_ngrams(doc.page_content) for doc in docs]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            openai_api_key=config.OPENAI_API_KEY,
            max_tokens=300,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_MAX_RETRIES,
        )

    def expand_query(self, query: str) -> str:
        try:
            messages = [
                (
                    "system",
                    "Rewrite the user's search query into concise Hindi/English keywords for retrieval. "
                    "Include direct translations when the query is Hinglish. Return only keywords.",
                ),
                ("human", query),
            ]
            response = self.llm.invoke(messages)
            expanded = response.content.strip()
            return f"{query}\n{expanded}" if expanded else query
        except (APIConnectionError, APIStatusError, AuthenticationError, RateLimitError, AttributeError):
            logger.info("Query expansion unavailable; using original query.", exc_info=True)
            return query

    def retrieve(self, query: str, k: int = 8) -> list[Document]:
        expanded_query = self.expand_query(query)
        query_tokens = tokenize(expanded_query)
        query_ngrams = char_ngrams(expanded_query)
        if not query_tokens and not query_ngrams:
            return self.docs[:k]

        scored_docs = []
        query_lower = expanded_query.lower()
        for index, doc in enumerate(self.docs):
            token_score = len(query_tokens.intersection(self.doc_tokens[index]))
            ngram_score = len(query_ngrams.intersection(self.doc_ngrams[index])) / 8
            phrase_score = doc.page_content.lower().count(query_lower) * 3
            scored_docs.append((token_score + ngram_score + phrase_score, index, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        selected = [doc for score, _, doc in scored_docs if score > 0][:k]
        return selected or self.docs[:k]

    def extractive_fallback(self, query: str, context_docs: list[Document]) -> str:
        query_tokens = tokenize(query)
        sentences = []
        for doc in context_docs:
            sentences.extend(re.split(r"(?<=[.!?])\s+|\n+", doc.page_content))

        ranked = []
        for sentence in sentences:
            clean_sentence = " ".join(sentence.split())
            if len(clean_sentence) < 20:
                continue
            score = len(query_tokens.intersection(tokenize(clean_sentence)))
            ranked.append((score, clean_sentence))

        ranked.sort(key=lambda item: item[0], reverse=True)
        best = [sentence for score, sentence in ranked if score > 0][:3]

        if best:
            return " ".join(best)
        if ranked:
            return " ".join(sentence for _, sentence in ranked[:2])
        return (
            "I can help with the information available in this link, such as its features, "
            "tech stack, models used, or setup details. Please ask a specific question."
        )

    def invoke(self, payload: dict) -> dict:
        query = payload["input"]
        context_docs = self.retrieve(query)
        context = "\n\n".join(doc.page_content for doc in context_docs)

        try:
            messages = self.prompt.format_messages(context=context, input=query)
            response = self.llm.invoke(messages)
            return {"answer": response.content, "context": context_docs}
        except (APIConnectionError, APIStatusError, AuthenticationError, RateLimitError):
            logger.error("OpenAI chat failed; using local extractive fallback.", exc_info=True)
            return {"answer": self.extractive_fallback(query, context_docs), "context": context_docs}


def setup_qa_chain(docs: list[Document]):
    return LocalRetrievalQA(docs)
