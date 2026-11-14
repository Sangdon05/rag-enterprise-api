import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from feature.chatbot.schemas import PolicyMetadata
from shared.core.exceptions import NotFoundException
from shared.core.logger import get_logger

logger = get_logger(__name__)


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_document(file_path: str) -> list[Document]:
    path = Path(file_path)
    if not path.exists():
        raise NotFoundException(f"지정한 문서 파일을 찾을 수 없습니다: {file_path}")
    loader = (
        PyPDFLoader(str(path))
        if path.suffix.lower() == ".pdf"
        else TextLoader(str(path), encoding="utf-8")
    )
    return loader.load()


def preview_chunking(
    file_path: str, small_size: int, large_size: int, overlap: int
) -> dict:
    raw_docs = load_document(file_path)
    seps = ["\n\n", "\n", ". ", " ", ""]

    small_chunks = RecursiveCharacterTextSplitter(
        chunk_size=small_size,
        chunk_overlap=overlap,
        separators=seps,
    ).split_documents(raw_docs)

    large_chunks = RecursiveCharacterTextSplitter(
        chunk_size=large_size,
        chunk_overlap=overlap,
        separators=seps,
    ).split_documents(raw_docs)

    return {
        "small_strategy": {
            "chunk_size": small_size,
            "count": len(small_chunks),
            "sample": small_chunks[0].page_content if small_chunks else "",
        },
        "large_strategy": {
            "chunk_size": large_size,
            "count": len(large_chunks),
            "sample": large_chunks[0].page_content if large_chunks else "",
        },
    }


def split_and_enrich(
    file_path: str, doc_id: str, department: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    raw_docs = load_document(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    enriched: list[Document] = []

    for i, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if not text:
            continue

        meta = PolicyMetadata(
            doc_id=doc_id,
            department=department,
            page=chunk.metadata.get("page", 0) + 1,
            chunk_index=i,
            content_hash=compute_sha256(text),
        )

        enriched.append(Document(page_content=text, metadata=meta.model_dump()))
    return enriched
