from langchain_core.documents import Document

from feature.recommend.bm25_retriever import search_bm25_catalog
from shared.core.config import settings
from shared.core.vector_store import get_vector_store


def search_vector_catalog(
    query: str,
    top_k: int = 8,
    category: str | None = None,
    max_budget: int | None = None,
) -> list[tuple[Document, float]]:
    store = get_vector_store(settings.RECOMMEND_COLLECTION)
    conditions: list[dict] = []
    if category:
        conditions.append({"category": {"$eq": category}})
    if max_budget is not None:
        conditions.append({"price": {"$lte": max_budget}})

    if len(conditions) == 1:
        filter_dict = conditions[0]
    elif len(conditions) > 1:
        filter_dict = {"$and": conditions}
    else:
        filter_dict = None

    return store.similarity_search_with_relevance_scores(
        query, k=top_k, filter=filter_dict
    )


def hybrid_rrf_search(
    keyword_query: str,
    semantic_query: str,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    rrf_k: int = 60,
    top_k: int = 5,
    category: str | None = None,
    max_budget: int | None = None,
) -> list[dict]:
    pool_size = top_k * 3
    bm25_hits = search_bm25_catalog(
        keyword_query, top_k=pool_size, category=category, max_budget=max_budget
    )
    vector_hit = search_vector_catalog(
        semantic_query, top_k=pool_size, category=category, max_budget=max_budget
    )

    registry: dict[str, dict] = {}

    for rank, (doc, raw_score) in enumerate(bm25_hits, start=1):
        pid = doc.metadata.get("product_id")
        if not pid:
            continue

        registry[pid] = {
            "document": doc,
            "bm25_rank": rank,
            "bm25_rw_score": round(float(raw_score), 4),
            "vector_rank": None,
            "vector_raw_score": 0.0,
            "rrf_score": bm25_weight * (1.0 / (rrf_k + rank)),
        }

    for rank, (doc, raw_score) in enumerate(vector_hit, start=1):
        pid = doc.metadata.get("product_id")
        if not pid:
            continue

        addition = vector_weight * (1.0 / (rrf_k + rank))
        if pid in registry:
            registry[pid]["vector_rank"] = rank
            registry[pid]["vector_raw_score"] = round(float(raw_score), 4)
            registry[pid]["rrf_score"] += addition
        else:
            registry[pid] = {
                "document": doc,
                "bm25_rank": None,
                "bm25_raw_score": 0.0,
                "vector_rak": rank,
                "vector_raw_score": round(float(raw_score), 4),
                "rrf_score": addition,
            }
    ranked_items = sorted(registry.values(), key=lambda x: x["rrf_score"], reverse=True)
    for item in ranked_items:
        item["rrf_score"] = round(item["rrf_score"], 6)

    return ranked_items[:top_k]
