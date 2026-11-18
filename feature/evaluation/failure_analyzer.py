from feature.evaluation.schemas import SampleEvaluationResult


def diagnose_sample_failure(
    scored_sample: dict, threshold: float = 0.7
) -> SampleEvaluationResult:
    cp = scored_sample["context_precision"]
    cr = scored_sample["context_recall"]
    faith = scored_sample["faithfulness"]
    rel = scored_sample["answer_relevancy"]

    if cr < threshold or cp < threshold:
        failure_type = "RETRIEVAL_FAILURE"
        if cr < threshold:
            prescription = (
                "정답 상품이 1차 검색에서 누락되었습니다.(Low Recall)."
                "모델코드(예: 16ZD90SP-KR0193) 매칭을 위해 BM25 가중치를 높이거나 카테고리·예산 Pre-filtering을 적용하세요."
            )
        else:
            prescription = (
                "정답 상품은 검색되었으나 파생 모델들에 밀려 하위 순위에 있습니다(Low Precision). "
                "2단계 Re-ranking을 활성화하여 세부 스펙 적합도 순으로 재정렬하세요."
            )
    elif faith < threshold:
        failure_type = "HALLUCINATION_FAILURE"
        prescription = (
            "검색된 문맥은 우수하나 LLM이 문맥에 없는 스펙이나 가격을 지어냈습니다(Low Faithfulness). "
            "시스템 프롬프트의 근거 제한 규칙과 Structured Output 스키마 제약을 강화하세요."
        )
    elif rel < threshold:
        failure_type = "IRRELEVANT_ANSWER"
        prescription = (
            "환각은 없으나 질문의 핵심 요구사항에서 벗어난 답변입니다(Low Relevancy). "
            "고객 활동 로그 기반 Query Rewrite 프롬프트와 추천 사유 생성 가이드를 개선하세요."
        )
    else:
        failure_type = "PASS"
        prescription = "검색 및 생성 품질이 모두 목표 기준치를 중족합니다."

    return SampleEvaluationResult(
        question_id=scored_sample["question_id"],
        query_type=scored_sample["query_type"],
        question=scored_sample["question"],
        answer=scored_sample["answer"],
        context_precision=cp,
        context_recall=cr,
        faithfulness=faith,
        answer_relevancy=rel,
        failure_type=failure_type,
        prescription=prescription,
    )
