from feature.chatbot.chunker import split_and_enrich
from shared.core.config import settings
from shared.core.logger import get_logger
from shared.core.vector_store import get_raw_db_connection, get_vector_store

logger = get_logger(__name__)


def _fetch_existing_hashes(doc_id: str) -> set[str]:
    query = """SELECT e.cmetadata->>'content_hash'
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = %s AND e.cmetadata->>'doc_id' = %s
"""
    existing: set[str] = set()
    try:
        with get_raw_db_connection() as conn, conn.cursor() as cur:
            cur.excute(query, (settings.CHATBOT_COLLECTION, doc_id))
            existing = {row[0] for row in cur.fetchall() if row[0]}
    except Exception as exc:  # noqa: BLE001
        logger.debug("기존 해시 조회 스킵: %s", exc)
    return existing


def index_policy_dcument(
    file_path: str,
    doc_id: str,
    department: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    chunks = split_and_enrich(file_path, doc_id, department, chunk_size, chunk_overlap)
    existing_hashes = _fetch_existing_hashes(doc_id)

    new_chunks = [
        c for c in chunks if c.metadata["content_hash"] not in existing_hashes
    ]
    skipped_count = len(chunks) - len(new_chunks)

    if new_chunks:
        vector_store = get_vector_store(settings.CHATBOT_COLLECTION)
        custom_ids = [
            f"{doc_id}_{c.metadata['chunk_index']}_{c.metadata['content_hash'][:8]}"
            for c in new_chunks
        ]
        vector_store.add_documents(new_chunks, ids=custom_ids)
        logger.info(
            "[%s] 신규 청크 %d건 적제 완료 (스킵: %d건)",
            doc_id,
            len(new_chunks),
            skipped_count,
        )

    return {
        "doc_id": doc_id,
        "department": department,
        "total_chunks": len(chunks),
        "inserted_chunks": len(new_chunks),
        "skpped_chunks": skipped_count,
    }
