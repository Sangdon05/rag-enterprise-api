from fastapi import APIRouter

from feature.chatbot.chunker import preview_chunking
from feature.chatbot.generator import ask_policy_chatbot
from feature.chatbot.indexer import index_policy_dcument
from feature.chatbot.schemas import (
    ChatbotAskRequest,
    ChunkPreviewReqeust,
    IndexDocumentRequest,
)
from shared.core.schemas import APIResponse

router = APIRouter(prefix="/v1/chatbot", tags=["[실습 1] 사내 규정 챗봇"])


@router.post("/chunks/preview", response_model=APIResponse[dict])
def preview_chunks_endpoint(req: ChunkPreviewReqeust):
    data = preview_chunking(
        req.file_path, req.small_chunk_size, req.large_chunk_size, req.chunk_overlap
    )
    return APIResponse.ok(data=data, message="청킹 전략 비교 실험이 완료되었습니다.")


@router.post("/documents/index", response_model=APIResponse[dict])
def index_document_endpoint(req: IndexDocumentRequest):
    data = index_policy_dcument(
        req.file_path, req.doc_id, req.department, req.chunk_size, req.chunk_overlap
    )
    return APIResponse.ok(
        data=data, message="사내 규정 문서 증분 인덱싱이 완료되었습니다."
    )


@router.post("/ask", response_model=APIResponse[dict])
def ask_chatbot_endpoint(req: ChatbotAskRequest):
    data = ask_policy_chatbot(
        req.question, req.department, req.top_k, req.score_threshold
    )
    return APIResponse.ok(data=data, message="사내 규정 질의응답 완료되었습니다.")
