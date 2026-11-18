import json
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from feature.evaluation.schemas import EvalQAItem
from feature.recommend.hybird_rrf import hybrid_rrf_search, search_vector_catalog
from feature.recommend.query_rewriter import rewrite_query_from_activity
from feature.recommend.reranker import rerank_and_compress_candidates
from feature.recommend.schemas import UserActivityLog
from shared.core.vector_store import get_llm

EVAL_DATASET_PATH = Path("data/eval_golden_dataset.json")

DEFAULT_GOLDEN_QA: list[dict] = [
    {
        "question_id": "QA-001",
        "query_type": "exact_keyword",
        "question": "모델코드 16ZD90SP-KR0193 노트북의 정확한 무게, GPU, RAM 용량과 가격을 알려줘",
        "ground_truth": "LG전자 그램 프로 16 (모델코드 16ZD90SP-KR0193, 상품코드 SKU-0193-16ZD90SP)은 무게 1.19kg, 코어 울트라7, RTX4070 8GB, 64GB RAM, 512GB NVMe 스펙을 갖추고 있으며 가격은 1,500,000원이다.",
        "category": "노트북",
        "max_budget": 1800000,
        "activity_keywords": ["16ZD90SP-KR0193", "그램 프로 16 64GB"],
    },
    {
        "question_id": "QA-002",
        "query_type": "single_hop",
        "question": "50만원 이하 예산으로 맥북 연결 및 KVM 스위치, 피벗 스탠드가 지원되는 LG 27인치 4K 모니터 추천해줘",
        "ground_truth": "LG전자 울트라파인 4K UHD (모델코드 27UQ850V-KR0002, 가격 460,000원) 또는 27UQ850V-KR0058 (가격 350,000원) 모델이 27인치 4K IPS Black 패널과 피벗/엘리베이션 스탠드를 지원하여 적합하다.",
        "category": "모니터",
        "max_budget": 500000,
        "activity_keywords": ["27UQ850V-KR0002", "27인치 4K 피벗 모니터"],
    },
    {
        "question_id": "QA-003",
        "query_type": "multi_hop",
        "question": "노트북에 케이블 하나만 꽂아서 듀얼 모니터와 유선랜을 연결할 18포트 확장 칼디짓 TS4 도킹스테이션 추천해줘",
        "ground_truth": "칼디짓 TS4 썬더볼트4 독 라인업(예: TS4-US-AMZ-KR0032, TS4-US-AMZ-KR0064 등)은 18포트 확장, 썬더볼트4/USB4 40Gbps, 90W~98W PD 충전 및 듀얼 4K 60Hz 출력을 지원하여 케이블 하나로 모든 주변기기를 연결할 수 있다.",
        "category": "도킹스테이션",
        "max_budget": 450000,
        "activity_keywords": ["TS4-US-AMZ", "칼디짓 TS4 18포트"],
    },
]


def load_or_create_golden_dataset() -> list[EvalQAItem]:
    EVAL_DATASET_PATH.parent.mkdir(exist_ok=True)
    if not EVAL_DATASET_PATH.exists():
        EVAL_DATASET_PATH.write_text(
            json.dumps(DEFAULT_GOLDEN_QA, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    raw_list = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))
    return [EvalQAItem(**item) for item in raw_list]


def collect_pipeline_outputs(pipeline_mode: str, top_k: int = 3) -> list[dict]:
    qa_items = load_or_create_golden_dataset()
    collected: list[dict] = []
    llm = get_llm()

    for item in qa_items:
        if pipeline_mode == "vector_only":
            vec_hits = search_vector_catalog(item.question, top_k=top_k)
            contexts = [doc.page_content for doc, _ in vec_hits]
            prompt = ChatPromptTemplate.from_template(
                "아래 [상품 문맥]만 참고하여 질문에 답변하세요.\n\n[상품 문맥]\n{context}\n\n[질문]: {question}"
            )
            answer = (prompt | llm | StrOutputParser()).invoke(
                {
                    "context": "\n\n".join(contexts) if contexts else "검색 결과 없음",
                    "question": item.question,
                }
            )
        else:
            log = UserActivityLog(
                recent_searches=item.activity_keywords,
                clicked_products=[],
                preferred_category=item.category,
                max_budget=item.max_bugget,
            )
            rewritten = rewrite_query_from_activity(item.question, log)
            candidates = hybrid_rrf_search(
                keyword_query=rewritten.keyword_query,
                semantic_query=rewritten.semantic_query,
                bm25_weight=0.6,
                vector_weight=0.4,
                top_k=top_k * 2,
                category=log.preferred_category,
                max_budget=log.max_budget,
            )
            reranked = rerank_and_compress_candidates(
                user_requst=item.question,
                intent_summary=rewritten.intent_summary,
                hybrid_candidates=candidates,
                top_k=top_k,
            )
            contexts = [c["document"].page_content for c in candidates[:top_k]]
            answer = (
                "\n".join(
                    f"- {r['title']} (모델코드: {r['model_code']}, 가격: {r['price']:,}원: {r['compressed_spec']} / 추천사유: {r['recommend_reason']}"
                    for r in reranked
                )
                or "조검에 맞는 추천 상품이 없습니다."
            )

        collected.append(
            {
                "question_id": item.question_id,
                "query_type": item.query_type,
                "question": item.question,
                "contexts": contexts if contexts else ["검색된 문맥 없음"],
                "answer": answer,
                "ground_truth": item.ground_truth,
            }
        )

    return collected
