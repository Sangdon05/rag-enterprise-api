from typing import cast

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from feature.chatbot.retriever import retrieve_policy_chunks
from feature.chatbot.schemas import CitedAnswerOutput
from shared.core.logger import get_logger
from shared.core.vector_store import get_llm

logger = get_logger(__name__)

SYSTEM_PROMPT = """당신은 사내 규정을 안내하는 AI 어시스턴트입니다.
반드시 아래 젝오된 [참고 규정 문맥]에 포함된 사실만을 근거로 답변하세요.

[작성 규칙]
1. 문맥에 근거가 전혀 없다면 추측하지 말고 answer에 "제공된 사내 규정에서 관련 내용을 찾을수 없습니다."라고 작성하고 cited_numbers는 빈 리스트([])로 반환하세요.
2. 답변 문장 끝에는 반드시 근거가 된 문맥 번호(예: [1], [2])를 표기하세요.
3. 실제 인용한 문맥 번호만 cited_numbers에 정수로 포함하세요.

[참고 규정 문맥]
{context}
"""


def _format_numbered_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"[{i}] 문서ID: {meta.get('doc_id')} | 부서: {meta.get('department')} | 페이지: {meta.get('page')}p"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def ask_policy_chatbot(
    question: str,
    department: str | None = None,
    top_k: int = 4,
    score_threshold: float = 0.35,
) -> dict:
    docs = retrieve_policy_chunks(question, department, top_k, score_threshold)
    if not docs:
        return {
            "question": question,
            "answer": "제공된 사내 규정에서 관련 내용을 찾을 수 없습니다.",
            "citations": [],
            "retrieved_count": 0,
        }

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{question}")]
    )
    structured_llm = get_llm().with_structured_output(CitedAnswerOutput)
    chain = prompt | structured_llm

    result = cast(
        CitedAnswerOutput,
        chain.invoke(
            {
                "context": _format_numbered_context(docs),
                "question": question,
            }
        ),
    )

    citations = []
    for num in result.cited_numbers:
        if 1 <= num <= len(docs):
            meta = docs[num - 1].metadata
            citations.append(
                {
                    "ref_number": num,
                    "doc_id": meta.get("doc_id"),
                    "department": meta.get("department"),
                    "page": meta.get("page"),
                }
            )

    return {
        "question": question,
        "answer": result.answer,
        "citations": citations,
        "retrieved_count": len(docs),
    }
