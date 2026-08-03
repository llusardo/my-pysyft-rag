"""Streamlit UI for the PySyft Docs Assistant RAG demo.

Single-page app: ask a question about PySyft/OpenMined docs, see the
generated answer and its source files. Uses GroqClient (free tier) for
generation — the zero-cost production path. Run locally with:
    streamlit run app.py
"""

import logging
import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from langfuse import observe

from rag.ingest import ingest_documents
from rag.llm_client import GroqClient
from rag.rag_service import RAGService

logger = logging.getLogger(__name__)

st.set_page_config(page_title="PySyft Docs Assistant")


@st.cache_resource
def get_rag_service() -> RAGService:
    """Build (once per app instance) the RAGService backing this app.

    Loads .env, builds/loads the Chroma collection (running ingestion on
    a cold start where chroma_data/ doesn't exist yet), and wires it up
    with GroqClient. Cached via st.cache_resource so this only runs once,
    not on every user interaction.

    Raises:
        RuntimeError: If GROQ_API_KEY is missing from the environment.
    """
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set — check your .env file")

    client = chromadb.PersistentClient(path="chroma_data")
    try:
        collection = client.get_collection("pysyft_docs")
    except Exception:
        with st.spinner("Setting up the docs index (first run only)..."):
            ingest_documents()
        collection = client.get_collection("pysyft_docs")

    groq_client = GroqClient(api_key=api_key)
    return RAGService(collection, groq_client)


@observe(name="rag_query")
def ask_rag(service: RAGService, question: str) -> dict:
    """Answer a question via RAGService, traced as a Langfuse observation.

    Kept outside RAGService so the service itself stays free of any
    observability-specific dependency (see Architecture Rules in CLAUDE.md).

    Args:
        service: The RAGService to query.
        question: The user's question.

    Returns:
        Same dict shape as RAGService.query(): "answer" and "sources".
    """
    return service.query(question)


st.title("PySyft Docs Assistant")
st.caption("Ask a question about PySyft / OpenMined docs.")

try:
    rag_service = get_rag_service()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.form("ask_form"):
    question = st.text_input("Your question")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.info("Type a question first.")
    else:
        with st.spinner("Thinking..."):
            try:
                result = ask_rag(rag_service, question)
            except Exception:
                logger.exception("Failed to answer question")
                st.error("Something went wrong answering that question — please try again.")
            else:
                st.subheader("Answer")
                st.write(result["answer"])

                sources = sorted({source["source"] for source in result["sources"]})
                if sources:
                    st.subheader("Sources")
                    for source in sources:
                        st.write(f"- {source}")
