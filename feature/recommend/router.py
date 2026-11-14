from fastapi import APIRouter

from feature.recommend.catalog_loader import sync_korean_products
from feature.recommend.hybird_rrf import hybrid_rrf_search
from feature.recommend.query_rewriter import rewrite_query_from_activity
from feature.recommend.reranker import rerank_and_compress_candidates
from feature.recommend.schemas import (
    CatalogSyncRequest,
    HybridSearchRequest,
    PersonalizedRecommendRequest,
)
from shared.core.schemas import APIResponse

router = APIRouter(
    prefix="/v1/recommend", tags=["[실습 2] 고객 활동 로그 기반 하이브리드 상품 추천"]
)


@router.post("/catalog/sync", response_model=APIResponse[dict])
def sync_catalog_endpoint(req: CatalogSyncRequest):
    data = sync_korean_products(file_path=req.file_path, batch_size=req.batch_size)
    return APIResponse.ok(
        data=data, message="한국어 상품 카탈로그(1,000건) 동기화가 완료되었습니다."
    )


@router.post("/search/hybrid", response_model=APIResponse[dict])
def hybrid_search_endpoint(req: HybridSearchRequest):
    results = hybrid_rrf_search(
        keyword_query=req.query,
        semantic_query=req.query,
        bm25_weight=req.bm25_weight,
        vector_weight=req.vector_wegith,
        rrf_k=req.rrf_k,
        top_k=req.top_k,
        category=req.category,
    )

    formatted = [
        {
            "product_id": r["document"].metadata.get("product_id"),
            "model_code": r["document"].metadata.get("model_code"),
            "title": r["document"].metadata.get("title"),
            "price": r["document"].metadata.get("price"),
            "bm25_rank": r["bm25_rank"],
            "vector_rank": r["vector_rank"],
            "rrf_score": r["rrf_score"],
        }
        for r in results
    ]

    return APIResponse.ok(data={"query": req.query, "results": formatted})


@router.post("/personalized", response_model=APIResponse[dict])
def personalized_recommend_endpoint(req: PersonalizedRecommendRequest):
    rewritten = rewrite_query_from_activity(req.user_request, req.activity_log)
    candidates = hybrid_rrf_search(
        keyword_query=rewritten.keyword_query,
        semantic_query=rewritten.semantic_query,
        top_k=req.top_k * 2,
        category=req.activity_log.preferred_category,
        max_budget=req.activity_log.max_budget,
    )
    recommendations = rerank_and_compress_candidates(
        user_requst=req.user_request,
        intent_summary=rewritten.intent_summary,
        hybrid_candidates=candidates,
        top_k=req.top_k,
    )
    return APIResponse.ok(
        data={
            "rewritten_queries": rewritten.model_dump(),
            "recommendations": recommendations,
        },
        message="고객 활동 로그 기반 하이브리드 상품 추처이 완료되었습니다.",
    )
