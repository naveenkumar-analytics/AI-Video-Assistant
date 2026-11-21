from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Naya package ho to wahi use karo, warna purana (dono chalte hain)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Project root mein vector_db (kisi bhi folder se run karo, same jagah rahega)
CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "vector_db")
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embeddings = None


def get_embeddings():
    """Embedding model sirf ek baar load hoga."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


def _reset_collection():
    """Purane video ka data hatao, taaki naye video ke saath mix na ho."""
    try:
        Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        ).delete_collection()
    except Exception:
        pass  # collection pehle se nahi thi, to koi dikkat nahi


def build_vector_store(transcript: str) -> Chroma:
    print("Building vector Store")

    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty, cannot build vector store.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    _reset_collection()

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"Vector store ready ({len(docs)} chunks).")
    return vector_store


def load_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )