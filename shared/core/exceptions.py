from typing import TypeVar

from fastapi import status

T = TypeVar("T")


class SharedException[T](Exception):
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        error_code: str = "BAD_REQUEST",
        errors: T | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.errors = errors
        super().__init__(message)


class BadRequestException(SharedException[T]):
    def __init__(
        self,
        message: str,
        error_code: str = "BAD_REQUEST",
        errors: T | None = None,
    ) -> None:
        super().__init__(message, status.HTTP_400_BAD_REQUEST, error_code, errors)


class NotFoundException(SharedException[None]):
    def __init__(
        self,
        message: str = "리소스를 찾을 수 없습니다.",
        error_code: str = "BAD_REQUEST",
    ) -> None:
        super().__init__(message, status.HTTP_404_NOT_FOUND, error_code, None)
