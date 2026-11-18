from typing import Literal

from pydantic import BaseModel, Field


class EvalQAItem(BaseModel):
    question_id: str = Field(description="평가 문항 고유 ID (예: QA-001)")
    query_type: Literal["exact_keyword", "single_hop", "multi_hop"] = Field(
        description="질의 유형 (고유모델코드/ 단순조건 / 복합스팩·예산추론"
    )
    question: str = Field(description="평가용 사용자 질문")
    ground_truth: str = Field(
        description="korean_products.json에 기반한 모범 정답 (Reference)"
    )
    category: str | None = Field(None, description="대상 카테고리")
    max_bugget: int | None = Field(None, description="예산 상환 (원)")
    activity_keywords: list[str] = Field(
        default_factory=list, description="고객 최근 검색어 로그"
    )


class RunEvaluationRequest(BaseModel):
    pipeline_mode: Literal["vector_only", "hybrid_rerank"] = Field(
        default="hybrid_rerank",
        description="평가 대상 파이프라인 모드 (Baseline 벡터 검색 vs 하이브리드+리랭킹)",
    )
    score_threshold_for_failure: float = Field(
        default=0.7, ge=0.0, le=1.0, description="실패 케이스로 분류할 임계점수"
    )


class RegressionCompareRequest(BaseModel):
    top_k: int = Field(default=3, ge=1, le=5, description="검색 및 추천 상위 K 개수")


class SampleEvaluationResult(BaseModel):
    question_id: str
    query_type: str
    question: str
    answer: str
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    failure_type: str = Field(
        description="PASS / RETRIEVAL_FAILURE / HYLLUCINATION / IRRELEVANT_ANSWER"
    )
    prescription: str = Field(description="자동 진단된 엔지니어링 처방전")
