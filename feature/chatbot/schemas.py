from pydantic import BaseModel, Field


class PolicyMetadata(BaseModel):
    doc_id: str = Field(description="원본 문서 고유 식별자 (예: hr_policy_2026")
    department: str = Field(description="소관 부서 (예: 인사팀, 정보보안팀)")
    page: int = Field(default=1, description="원본 문서 페이지 번호")
    chunk_index: int = Field(description="문서 내 청크 순번")
    content_hash: str = Field(description="증분 인덱싱 판별을 위한 SHA-256 해시값")


class ChunkPreviewReqeust(BaseModel):
    file_path: str = Field(default="data/hr_policy.txt", description="대산 문서 경로")
    small_chunk_size: int = Field(default=100, ge=50, le=1000)
    large_chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)


class IndexDocumentRequest(BaseModel):
    file_path: str = Field(
        default="data/hr_policy.txt", description="인덱싱할 문서 경로"
    )
    doc_id: str = Field(default="data_policy_2026", description="문서 고유 식별자")
    department: str = Field(default="인사팀", description="소관 부서명")
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)


class ChatbotAskRequest(BaseModel):
    question: str = Field(default="재택근무 코어타임은 몇시부터 몇시까지인가요?")
    department: str | None = Field(default="인사팀", description="부서 필터 (선택)")
    top_k: int = Field(default=4, ge=1, le=10)
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class CitedAnswerOutput(BaseModel):
    answer: str = Field(
        description="제공된 문맥만을 근거로 작성된 답변 (문장 끝에 [1], [2] 인용 표기)"
    )
    cited_numbers: list[int] = Field(
        default_factory=list, description="실제 인용된 문맥 번호 배열"
    )
