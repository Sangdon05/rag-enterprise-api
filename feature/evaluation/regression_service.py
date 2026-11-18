from feature.evaluation.dataset_builder import collect_pipeline_outputs
from feature.evaluation.failure_analyzer import diagnose_sample_failure
from feature.evaluation.metrics_runner import run_ragas_metrics


def _summarize_averages(results: list[dict]) -> dict[str, float]:
    count = len(results) or 1
    return {
        "context_precision": round(
            sum(r["context_precision"] for r in results) / count, 4
        ),
        "context_recall": round(sum(r["context_recall"] for r in results) / count, 4),
        "faithfulness": round(sum(r["faithfulness"] for r in results) / count, 4),
        "answer_relevancy": round(
            sum(r["answer_relevancy"] for r in results) / count, 4
        ),
    }


def evaluate_single_pipeline(pipeline_mode: str, threshold: float = 0.7) -> dict:
    samples = collect_pipeline_outputs(pipeline_mode=pipeline_mode)
    scored = run_ragas_metrics(samples)
    diagnosed = [
        diagnose_sample_failure(s, threshold=threshold).model_dump() for s in scored
    ]

    return {
        "pipeline_mode": pipeline_mode,
        "average_metrics": _summarize_averages(scored),
        "sample_details": diagnosed,
    }


def run_regression_comparison(top_k: int = 3) -> dict:
    base_sample = collect_pipeline_outputs("vector_only", top_k=top_k)
    hybrid_sample = collect_pipeline_outputs("hybrid_rerank", top_k=top_k)

    base_scored = run_ragas_metrics(base_sample)
    hybrid_scored = run_ragas_metrics(hybrid_sample)

    base_avg = _summarize_averages(base_scored)
    hybrid_avg = _summarize_averages(hybrid_scored)

    improvement_delta = {
        metric: round(hybrid_avg[metric] - base_avg[metric], 4) for metric in base_avg
    }

    return {
        "baseline_vector_only": base_avg,
        "improved_hybrid_rerank": hybrid_avg,
        "improvement_delta": improvement_delta,
        "regression_passed": all(
            delta >= -0.02 for delta in improvement_delta.values()
        ),
    }
