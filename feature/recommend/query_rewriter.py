from typing import cast

from langchain_core.prompts import ChatPromptTemplate

from feature.recommend.schemas import RewrittenQueryOutput, UserActivityLog
from shared.core.vector_store import get_llm

REWRITE_PROMPT = """당신은 한국어 IT·가전 이커머스 검색 최적화 AI입니다.
고객의 [현재 요청]과 [최근 쇼핑 활동 로그]를 종합 분석하여 하이브리드 검색기에 투입할 최적의 이원화 질의를 생성하세요.

[작성 규칙]
1. keyword_query: BM25 검색기가 상품명과 스펙에서 정확히 매칭할 수 있도록 활동 로그에 등장한 모델코드(예: 16ZD90SP, GU605MV), 브랜드, 하드웨어 스펙(RTX4060, RTX4070, 32GB, 64GB, 1.19kg 등) 핵심 명사만 공백으로 구분해 작성하세요.
2. semantic_query: PGVector가 상품 설명 및 구매자 리뷰와 의미적으로 매칭할 수 있도록 고객이 원하는 사용 시나리오(예: 외부 미팅 백팩 휴대, 도커 컨테이너 구동, 저소음 코딩 등)를 서술형 문장으로 작성하세요.
"""


def rewrite_query_from_activity(
    user_request: str, activity_log: UserActivityLog
) -> RewrittenQueryOutput:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_PROMPT),
            (
                "human",
                (
                    "[현재 요청]: {user_request}\n"
                    "[최근 검색어]: {recent_searches}\n"
                    "[최근 클릭 상품]: {clicked_products}\n"
                    "[관심 카테고리]: {preferred_category}\n"
                    "[최대 예산]: {max_budget}원"
                ),
            ),
        ]
    )
    chain = prompt | get_llm().with_structured_output(RewrittenQueryOutput)
    return cast(
        RewrittenQueryOutput,
        chain.invoke(
            {
                "user_request": user_request,
                "recent_searches": ", ".join(activity_log.recent_searches) or "없음",
                "clicked_products": ", ".join(activity_log.clicked_products) or "없음",
                "preferred_category": activity_log.preferred_category or "전체",
                "max_budget": activity_log.max_budget or "제한 없음",
            }
        ),
    )


async def require_owner(): ...
