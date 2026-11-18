import sys
import types

# [호환성 패치] langchain-community 1.x에서 제거된 vertexai 모듈을 ragas가 참조할 때 발생하는 에러 방지
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    _answer_relevance,
    _context_precision,
    _context_recall,
    _faithfulness,
)

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
    evaluator_llm = LangchainLLMWrapper(get_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings)

    metrics = [_context_precision, _context_recall, _faithfulness, _answer_relevance]

    eval_result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )
    scores_df = eval_result.to_pandas()
    enriched_results: list[dict] = []
