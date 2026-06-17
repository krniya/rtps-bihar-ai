"""Conversational RAG chain for the Bihar RTPS portal assistant."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic

from retrieval.retrieve_docs import get_pgvector_retriever

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SYSTEM_PROMPT = """You are an official assistant for the Bihar RTPS portal. Use the following retrieved contexts to answer the citizen's query accurately in the language they used (English or Hindi). If the answer cannot be found in the context, politely say you do not have that specific information. Always cite the document source metadata at the end of your answer.

{context}

Question: {question}
Answer:"""

DOCUMENT_PROMPT = PromptTemplate(
    input_variables=["page_content", "source_url", "department", "service_type"],
    template=(
        "Source URL: {source_url}\n"
        "Department: {department}\n"
        "Service Type: {service_type}\n"
        "Content:\n{page_content}"
    ),
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT,
)


def build_conversational_rag_chain(
    *,
    top_k: int = 4,
    department_filter: str | None = None,
    model: str = "claude-sonnet-4-6",
    temperature: float = 0.2,
) -> ConversationalRetrievalChain:
    """Build a ConversationalRetrievalChain wired to pgvector and chat memory."""

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    llm = ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=anthropic_api_key,
    )

    retriever = get_pgvector_retriever(
        department_filter=department_filter,
        top_k=top_k,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": QA_PROMPT,
            "document_prompt": DOCUMENT_PROMPT,
        },
        verbose=False,
    )


def ask_question(chain: ConversationalRetrievalChain, question: str) -> dict:
    """Run one conversational turn and return the chain response."""

    return chain.invoke({"question": question})
