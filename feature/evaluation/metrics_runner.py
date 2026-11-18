import sys
import types
from typing import cast

# [호환성 패치] langchain-community 1.x에서 제거된 vertexai 모듈을 ragas가 참조할 때 발생하는 에러 방지
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from shared.core.vector_store import get_embeddings, get_llm


def run_ragas_metrics(collected_samples: list[dict]) -> list[dict]:
    ragas_dataset = Dataset.from_dict(
        {
            "question": [s["question"] for s in collected_samples],
            "contexts": [s["contexts"] for s in collected_samples],
            "answer": [s["answer"] for s in collected_samples],
            "ground_truth": [s["ground_truth"] for s in collected_samples],
        }
    )
    # gpt-5/o-series 등 reasoning 계열 모델은 temperature=1만 지원하므로,
    # ragas가 temperature(0.01/0.3)와 n을 강제 설정하지 않도록 bypass 옵션을 사용
    evaluator_llm = LangchainLLMWrapper(
        get_llm(), bypass_temperature=True, bypass_n=True
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]

    eval_result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )
    scores_df = cast(EvaluationResult, eval_result).to_pandas()
    enriched_results: list[dict] = []

    for i, sample in enumerate(collected_samples):
        row = scores_df.iloc[i]
        enriched_results.append(
            {
                **sample,
                "context_precision": round(
                    float(row.get("context_precision", 0.0) or 0.0), 4
                ),
                "context_recall": round(
                    float(row.get("context_recall", 0.0) or 0.0), 4
                ),
                "faithfulness": round(float(row.get("faithfulness", 0.0) or 0.0), 4),
                "answer_relevancy": round(
                    float(row.get("answer_relevancy", 0.0) or 0.0), 4
                ),
            }
        )

    return enriched_results
