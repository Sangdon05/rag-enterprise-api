from pydantic import BaseModel, Field


class KoreanProductMetadata(BaseModel):
    product_id: str = Field(description="상품 고유 SKU (예: SKU-0193-16ZD90SP")
    model_code: str = Field(description="제조사 세부 모델코드 (예: 16ZD90SP-KR0193")
    title: str = Field(description="한국어 상품명 전체")
    brand: str = Field(description="브랜드명 (예: LG전자, ASUS, Apple, DELL)")
    category: str = Field(
        description="카테고리 (예: 노트북, 모니터, 키보드, 마우스, 오디오, 테블릿, 외장SSD, 도킹스테이션)"
    )
    price: int = Field(description="판매 가격 (원)")


class CatalogSyncRequest(BaseModel):
    file_path: str = Field(
        default="data/korean_products.json", description="적재할 상품 JSON 파일 경로"
    )
    batch_size: int = Field(
        default=100, ge=10, le=500, description="임베딩 API 배치 처리 단위"
    )


class UserActivityLog(BaseModel):
    recent_searches: list[str] = Field(
        default=["16ZD90SP-KR0193", "RTX4070 가벼운 노트북", "64GB 램 랩탑"],
        description="최근 검색 키워드 목록",
    )
    clicked_products: list[str] = Field(
        default=[
            "LG전자 그램 프로 16 [16ZD90SP-KR0193]",
            "ASUS ROG 제피러스 G16 [GU605MV-KR0057]",
        ],
        description="최근 클릭하여 상ㅅ페이지를 조회한 상품명 목록",
    )
    preferred_category: str | None = Field(
        default="노트북", description="관심 카테고리 필터 (선택)"
    )
    max_budget: int | None = Field(
        default=1800000, description="고객 선호 최대 예산 (원)"
    )


class HybridSearchRequest(BaseModel):
    query: str = Field(
        default="16ZD90SP-KR0193 코어 울트라7 64GB 노트북", description="검색 질의"
    )
    category: str | None = Field(default="노트북", description="카테고리 필터 (선태)")
    bm25_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="BM25 가중치)")
    vector_wegith: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Vector 가중치"
    )
    rrf_k: int = Field(default=60, ge=1, le=100, description="RRF 완충 상수 K")
    top_k: int = Field(default=5, ge=1, le=10, description="상위 선택 개수 K")


class PersonalizedRecommendRequest(BaseModel):
    user_request: str = Field(
        default="외부 미팅과 출장이 잦은데 백팩에 넣고 다니면서 도커 컨테이너 여러 개랑 로컬 LLM 돌릴 노트북 추천해줘",
        description="고객의 현재 자연어 요청",
    )
    activity_log: UserActivityLog = Field(default_factory=UserActivityLog)
    top_k: int = Field(default=3, ge=1, le=5, description="상위 선택 개수 K")


class RewrittenQueryOutput(BaseModel):
    keyword_query: str = Field(
        description="BM25 검색에 최적화된 모델코드/브랜드/스팩 중심 공백 구분 키워드"
    )
    semantic_query: str = Field(
        description="PGVector 검색에 최적화된 사용 목적/리뷰 감성 중심 서술형 질의"
    )
    intent_summary: str = Field(
        description="활동 로그와 현재 요청을 통합한 핵심 구매 의도 1줄 요약"
    )


class RerankedItem(BaseModel):
    doc_index: int = Field(description="후보 상품 번호 (1부터 시작)")
    relevance_score: float = Field(
        description="고객 구매 의도와의 적합도 점수 (0.0 ~ 1.0)"
    )
    compressed_spec: str = Field("원본 스팩과 설명에서 추출한 핵심 요약 (1문장)")
    recommend_reason: str = Field("고객 활동 로그 및 리뷰와 연결된 맞춤형 추천 사유")


class RerankOutput(BaseModel):
    selected_items: list[RerankedItem] = Field(
        description="적합도 순으로 재정렬된 최종 추처 상품 목록"
    )
