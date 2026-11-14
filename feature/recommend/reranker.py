from typing import cast

from langchain_core.prompts import ChatPromptTemplate

from feature.recommend.schemas import RerankOutput
from shared.core.vector_store import get_llm

RERANK_PROMPT = """당신은 초개인화 IT·가전 상품 리랭커(Re-ranker) 및 맥락 압축기입니다.
1차 하이브리드 검색으로 추출된 [후보 상품 목록]을 검토하여, [고객 요청 및 구매 의도]에 가장 부합하는 순서로 재정렬하세요.

[수행 규칙]
1. 후보 상품들의 세부 스펙(무게, RAM 용량, GPU, 가격)과 구매자 리뷰가 고객의 요청 조건과 일치하는지 엄격히 대조하세요.
2. compressed_spec에는 해당 상품의 핵심 하드웨어 스펙(무게/CPU/GPU/RAM/저장장치)만 1줄로 깔끔하게 요약하세요.
3. recommend_reason에는 고객의 최근 활동 로그(클릭/검색 이력) 및 실제 구매자 리뷰 내용을 인용하여 왜 이 모델코드 상품이 최적의 선택인지 설득력 있게 작성하세요.
"""


def rerank_and_compress_candidates(
    user_requst: str,
    intent_summary: str,
    hybrid_candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    if not hybrid_candidates:
        return []

    formatted = []
    for i, item in enumerate(hybrid_candidates, start=1):
        doc = item["document"]
        meta = doc.metadata
        formatted.append(
            f"[{i}] 상품명: {meta.get('title')} | 모델코드: {meta.get('model_code')} | 가격: {meta.get('price'):,}원\n{doc.page_content}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RERANK_PROMPT),
            (
                "human",
                (
                    "[고객 요청]: {user_request}\n"
                    "[활동 로그 기반 구매 의도]: {intent_summary}\n"
                    "[후보 상품 목록]\n{candidates_text}"
                ),
            ),
        ]
    )
    chain = prompt | get_llm().with_structured_output(RerankOutput)
    reranked = cast(
        RerankOutput,
        chain.invoke(
            {
                "user_request": user_requst,
                "intent_summary": intent_summary,
                "candidates_text": "\n\n".join(formatted),
            }
        ),
    )

    final_items: list[dict] = []
    for sel in reranked.selected_items[:top_k]:
        i = sel.doc_index - 1
        if 0 <= i < len(hybrid_candidates):
            cand = hybrid_candidates[i]
            meta = cand["document"].metadata
            final_items.append(
                {
                    "product_id": meta.get("product_id"),
                    "model_code": meta.get("model_code"),
                    "title": meta.get("title"),
                    "brand": meta.get("brand"),
                    "category": meta.get("category"),
                    "price": meta.get("price"),
                    "relevance_score": sel.relevance_score,
                    "rrf_score": cand["rrf_score"],
                    "compressed_spec": sel.compressed_spec,
                    "recommend_reason": sel.recommend_reason,
                }
            )
    return final_items
