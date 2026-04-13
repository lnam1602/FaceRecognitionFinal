"""
Pydantic v2 schemas cho tất cả request/response của API.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from server.config import DEFAULT_THRESHOLD, MAX_REGISTER_IMAGES


# ──────────────────────────────────────────────
# REQUEST MODELS
# ──────────────────────────────────────────────


class RegisterRequest(BaseModel):
    person_id: str = Field(..., min_length=1, max_length=50, description="Mã định danh người dùng, vd: SV001")
    name: str = Field(..., min_length=1, max_length=100, description="Tên hiển thị, vd: Nguyen Van A")
    images: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_REGISTER_IMAGES,
        description=f"Danh sách ảnh base64 (JPEG/PNG). Tối thiểu 1, tối đa {MAX_REGISTER_IMAGES}. Khuyến nghị 3-10 ảnh.",
    )

    @model_validator(mode="after")
    def check_images_not_empty(self) -> "RegisterRequest":
        for i, img in enumerate(self.images):
            if not img.strip():
                raise ValueError(f"images[{i}] không được rỗng")
        return self


class RecognizeRequest(BaseModel):
    image: str = Field(..., description="Ảnh base64 (JPEG/PNG) cần nhận diện")
    threshold: float = Field(
        DEFAULT_THRESHOLD,
        ge=0.1,
        le=1.0,
        description="Ngưỡng cosine distance. Thấp hơn = chặt hơn. Mặc định 0.4.",
    )


# ──────────────────────────────────────────────
# RESPONSE MODELS
# ──────────────────────────────────────────────


class RegisterResponse(BaseModel):
    status: Literal["ok", "error"]
    person_id: str
    name: str
    embedding: list[float] = Field(description="512-dim ArcFace embedding. App lưu lại để cache offline.")
    faces_detected: int = Field(description="Số ảnh detect được khuôn mặt thành công")
    faces_failed: int = Field(description="Số ảnh không detect được khuôn mặt")


class RecognizeResponse(BaseModel):
    status: Literal["recognized", "unknown", "no_face"]
    person_id: Optional[str] = None
    name: Optional[str] = None
    confidence: Optional[float] = Field(None, description="1 - distance, dễ đọc hơn cho client")
    distance: Optional[float] = Field(None, description="Cosine distance (thấp hơn = giống hơn)")


class PersonInfo(BaseModel):
    person_id: str
    name: str
    registered_at: str


class PersonsListResponse(BaseModel):
    persons: list[PersonInfo]


class DeleteResponse(BaseModel):
    status: Literal["ok", "not_found"]
    deleted: Optional[str] = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    registered_persons: int


class ErrorResponse(BaseModel):
    status: Literal["error"]
    detail: str
