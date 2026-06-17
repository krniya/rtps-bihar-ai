"""Streamlit chat UI for the Bihar RTPS conversational RAG assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chain import build_conversational_rag_chain


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = build_conversational_rag_chain()


def format_sources(source_documents) -> str:
    if not source_documents:
        return ""

    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for doc in source_documents:
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        source_key = (
            metadata.get("source_url", ""),
            metadata.get("department", ""),
            metadata.get("service_type", ""),
        )
        if source_key in seen:
            continue
        seen.add(source_key)
        lines.append(
            f"- {metadata.get('department', 'Unknown department')} | "
            f"{metadata.get('service_type', 'Unknown service')} | "
            f"{metadata.get('source_url', 'No URL')}"
        )

    if not lines:
        return ""

    return "\n\n**Retrieved sources:**\n" + "\n".join(lines)


def main() -> None:
    st.set_page_config(
        page_title="Bihar RTPS Assistant",
        page_icon="🏛️",
        layout="centered",
    )

    st.title("Bihar RTPS Portal Assistant")
    st.caption(
        "Ask questions about Bihar RTPS services in English or Hindi. "
        "Answers are grounded in indexed portal documents."
    )

    try:
        init_session_state()
    except ValueError as exc:
        st.error(str(exc))
        st.info("Copy `.env.example` to `.env` and set ANTHROPIC_API_KEY and OPENAI_API_KEY.")
        return
    except Exception as exc:
        st.error(f"Failed to initialize the assistant: {exc}")
        return

    with st.sidebar:
        st.subheader("Session")
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.chain = build_conversational_rag_chain()
            st.rerun()

        st.markdown(
            "This assistant uses pgvector retrieval and keeps conversation context "
            "with `ConversationBufferMemory`."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about RTPS services, certificates, licences..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and preparing answer..."):
                try:
                    result = st.session_state.chain.invoke({"question": prompt})
                    answer = result.get("answer", "Sorry, I could not generate a response.")
                    sources = format_sources(result.get("source_documents"))
                    full_response = answer + sources
                except Exception as exc:
                    full_response = (
                        "Sorry, something went wrong while processing your question. "
                        f"Please try again.\n\n`{exc}`"
                    )

            st.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )


if __name__ == "__main__":
    main()
