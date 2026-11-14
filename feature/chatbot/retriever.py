from langchain_core.documents import Document

from shared.core.config import settings
from shared.core.logger import get_logger
from shared.core.vector_store import get_vector_store

logger = get_logger(__name__)


def retrieve_policy_chunks(
    question: str,
    department: str | None = None,
    top_k: int = 4,
    score_threshold: float = 0.35,
) -> list[Document]:
    vector_store = get_vector_store(settings.CHATBOT_COLLECTION)

    search_kwargs: dict = {"k": top_k, "score_threshold": score_threshold}
    if department:
        search_kwargs["filter"] = {"department": {"$eq": department}}

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs,
    )
    docs = retriever.invoke(question)
    logger.info(
        "질의 검색 완료: question='%s', department='%s' -> %d건",
        question,
        department or "전체",
        len(docs),
    )
    return docs
