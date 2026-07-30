from dotenv import load_dotenv
import os
import chromadb
from rag.rag_service import RAGService
from rag.llm_client import AnthropicClient

if __name__ == "__main__":
    load_dotenv()

    client = chromadb.PersistentClient(path="chroma_data")
    collection = client.get_collection("pysyft_docs")

    llm = AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])
    service = RAGService(collection, llm)

    result = service.query("What are the core principles of Syft?")
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES:", [s["source"] for s in result["sources"]])
