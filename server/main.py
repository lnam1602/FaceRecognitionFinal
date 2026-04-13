"""
FastAPI application — điểm vào chính của server nhận diện khuôn mặt.

Chạy:
    uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader

from server.config import API_KEY
from server.database import (
    build_embeddings_matrix,
    delete_person,
    get_all_persons,
    load_db,
    upsert_person,
)
from server.face_service import FaceService
from server.models import (
    DeleteResponse,
    ErrorResponse,
    HealthResponse,
    PersonsListResponse,
    RecognizeRequest,
    RecognizeResponse,
    RegisterRequest,
    RegisterResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# API Key security
# ──────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key không hợp lệ hoặc thiếu header X-API-Key")
    return api_key


# ──────────────────────────────────────────────
# Lifespan — khởi tạo model và DB khi startup
# ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chạy 1 lần khi server khởi động. Tải model và database vào app.state."""
    logger.info("=== Server đang khởi động ===")

    # Tải InsightFace model
    face_service = FaceService()
    app.state.face_service = face_service

    # Load face database từ Drive / local
    db = load_db()
    app.state.db = db

    # Build ma trận embedding vectorized
    matrix, ids = build_embeddings_matrix(db)
    app.state.embeddings_matrix = matrix
    app.state.person_ids = ids

    logger.info("=== Server sẵn sàng. Đã load %d người. ===", len(db))
    yield
    logger.info("=== Server đang tắt ===")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────

app = FastAPI(
    title="Face Recognition Server",
    description=(
        "Server nhận diện khuôn mặt dùng InsightFace (ArcFace) chạy trên Google Colab. "
        "Phục vụ app điểm danh với 2 tính năng: đăng ký khuôn mặt và nhận diện."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def _rebuild_matrix(request: Request) -> None:
    """Rebuild ma trận embedding sau khi DB thay đổi."""
    matrix, ids = build_embeddings_matrix(request.app.state.db)
    request.app.state.embeddings_matrix = matrix
    request.app.state.person_ids = ids


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health(request: Request):
    """Kiểm tra trạng thái server — không cần API key."""
    return HealthResponse(
        status="ok",
        model_loaded=hasattr(request.app.state, "face_service"),
        registered_persons=len(request.app.state.db),
    )


@app.post(
    "/register",
    response_model=RegisterResponse,
    responses={422: {"model": ErrorResponse}},
    tags=["Face"],
)
async def register(
    body: RegisterRequest,
    request: Request,
    _key: str = Security(verify_api_key),
):
    """
    Đăng ký khuôn mặt mới (hoặc cập nhật nếu person_id đã tồn tại).

    Gửi 3-10 ảnh để đạt độ chính xác tốt nhất.
    Server trả về embedding 512-dim — app nên lưu lại để cache offline.
    """
    face_service: FaceService = request.app.state.face_service

    embeddings = []
    failed = 0

    for b64_img in body.images:
        img = face_service.decode_image(b64_img)
        if img is None:
            failed += 1
            continue

        emb = face_service.get_embedding(img)
        if emb is None:
            failed += 1
            logger.debug("Không detect được khuôn mặt trong 1 ảnh của %s", body.person_id)
        else:
            embeddings.append(emb)

    if not embeddings:
        raise HTTPException(
            status_code=422,
            detail="Không detect được khuôn mặt trong bất kỳ ảnh nào. Hãy chụp lại với ánh sáng đủ và khuôn mặt rõ ràng.",
        )

    # Tính embedding đại diện bằng cách average
    representative_emb = face_service.average_embeddings(embeddings)

    # Ghi vào DB (in-memory + Drive)
    registered_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    upsert_person(
        request.app.state.db,
        person_id=body.person_id,
        name=body.name,
        embedding=representative_emb,
        registered_at=registered_at,
    )

    # Rebuild ma trận vectorized
    _rebuild_matrix(request)

    logger.info(
        "Đã đăng ký %s (%s): %d/%d ảnh thành công",
        body.person_id,
        body.name,
        len(embeddings),
        len(body.images),
    )

    return RegisterResponse(
        status="ok",
        person_id=body.person_id,
        name=body.name,
        embedding=representative_emb.tolist(),
        faces_detected=len(embeddings),
        faces_failed=failed,
    )


@app.post(
    "/recognize",
    response_model=RecognizeResponse,
    tags=["Face"],
)
async def recognize(
    body: RecognizeRequest,
    request: Request,
    _key: str = Security(verify_api_key),
):
    """
    Nhận diện khuôn mặt từ 1 ảnh.

    Trả về `status`:
    - `recognized`: nhận ra người với `person_id` và `name`
    - `unknown`: có mặt nhưng không khớp ai trong DB
    - `no_face`: không phát hiện khuôn mặt trong ảnh
    """
    face_service: FaceService = request.app.state.face_service

    img = face_service.decode_image(body.image)
    if img is None:
        raise HTTPException(status_code=422, detail="Không decode được ảnh. Hãy kiểm tra định dạng base64.")

    emb = face_service.get_embedding(img)
    if emb is None:
        return RecognizeResponse(status="no_face")

    pid, name, distance = face_service.find_best_match(
        query_embedding=emb,
        embeddings_matrix=request.app.state.embeddings_matrix,
        person_ids=request.app.state.person_ids,
        db=request.app.state.db,
        threshold=body.threshold,
    )

    if pid is not None:
        return RecognizeResponse(
            status="recognized",
            person_id=pid,
            name=name,
            confidence=round(1.0 - distance, 4),
            distance=round(distance, 4),
        )

    return RecognizeResponse(
        status="unknown",
        distance=round(distance, 4),
    )


@app.get(
    "/persons",
    response_model=PersonsListResponse,
    tags=["Persons"],
)
async def list_persons(
    request: Request,
    _key: str = Security(verify_api_key),
):
    """Lấy danh sách tất cả người đã đăng ký (không kèm embedding vector)."""
    persons = get_all_persons(request.app.state.db)
    return PersonsListResponse(persons=persons)


@app.delete(
    "/persons/{person_id}",
    response_model=DeleteResponse,
    tags=["Persons"],
)
async def remove_person(
    person_id: str,
    request: Request,
    _key: str = Security(verify_api_key),
):
    """Xóa 1 người khỏi database. Dùng để sửa lỗi đăng ký nhầm."""
    deleted = delete_person(request.app.state.db, person_id)
    if not deleted:
        return DeleteResponse(status="not_found")

    _rebuild_matrix(request)
    logger.info("Đã xóa person_id: %s", person_id)
    return DeleteResponse(status="ok", deleted=person_id)
