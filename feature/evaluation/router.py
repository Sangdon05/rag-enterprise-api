from fastapi import APIRouter

from feature.evaluation.dataset_builder import load_or_create_golden_dataset
from feature.evaluation.regression_service import (
    evaluate_single_pipeline,
    run_regression_comparison,
)
from feature.evaluation.schemas import RegressionCompareRequest, RunEvaluationRequest
from shared.core.schemas import APIResponse

router = APIRouter(
    prefix="/v1/evaluations", tags=["[실습 3] RAGAS 정량 평가 및 회귀 검증"]
)


@router.get("/dataset", response_model=APIResponse[list])
def get_golden_dataset_endpoint():
    dataset = [item.model_dump() for item in load_or_create_golden_dataset()]
    return APIResponse.ok(
        data=dataset, message="Golden QA 평가셋 조회가 완료되었습니다."
    )


@router.post("/run", response_model=APIResponse[dict])
def run_evaluation_endpoint(req: RunEvaluationRequest):
    data = evaluate_single_pipeline(req.pipeline_mode, req.score_threshold_for_failure)
    return APIResponse.ok(
        data=data, message="RAGS 정량 평가 및 실패 원일 진단이 완료되었습니다."
    )


@router.post("/regression", response_model=APIResponse[dict])
def run_regression_endpoint(req: RegressionCompareRequest):
    data = run_regression_comparison(top_k=req.top_k)
    return APIResponse.ok(
        data=data, message="파이프라인 개선 전후 회귀 검증 비교가 완료되었습니다."
    )
