"""
Quản lý dữ liệu khuôn mặt — đọc/ghi face_db.json trên Google Drive.

Schema face_db.json:
{
  "SV001": {
    "name": "Nguyen Van A",
    "embedding": [0.023, -0.417, ...],   // 512 floats
    "registered_at": "2026-04-13T10:30:00"
  },
  ...
}
"""

import json
import os
import logging
from typing import Optional

import numpy as np

from server.config import DRIVE_DB_PATH, LOCAL_DB_PATH

logger = logging.getLogger(__name__)


def _get_db_path() -> str:
    """Trả về đường dẫn DB phù hợp: Drive nếu mount, ngược lại local."""
    drive_dir = os.path.dirname(DRIVE_DB_PATH)
    if os.path.isdir(drive_dir):
        return DRIVE_DB_PATH
    logger.warning(
        "Google Drive chưa mount (%s không tồn tại). Dùng local path: %s",
        drive_dir,
        LOCAL_DB_PATH,
    )
    return LOCAL_DB_PATH


def load_db() -> dict:
    """
    Load face database từ file JSON.
    Trả về dict rỗng nếu file chưa tồn tại.
    """
    path = _get_db_path()
    if not os.path.exists(path):
        logger.info("DB chưa tồn tại tại %s, khởi tạo rỗng.", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Chuyển embedding list → numpy array để tính toán nhanh hơn
    db = {}
    for person_id, record in raw.items():
        db[person_id] = {
            "name": record["name"],
            "embedding": np.array(record["embedding"], dtype=np.float32),
            "registered_at": record["registered_at"],
        }

    logger.info("Loaded %d người từ %s", len(db), path)
    return db


def save_db(db: dict) -> None:
    """
    Ghi face database xuống file JSON.
    numpy array được chuyển thành list trước khi serialize.
    """
    path = _get_db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    serializable = {}
    for person_id, record in db.items():
        emb = record["embedding"]
        serializable[person_id] = {
            "name": record["name"],
            "embedding": emb.tolist() if isinstance(emb, np.ndarray) else emb,
            "registered_at": record["registered_at"],
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d người vào %s", len(db), path)


def upsert_person(
    db: dict,
    person_id: str,
    name: str,
    embedding: np.ndarray,
    registered_at: str,
) -> None:
    """
    Thêm hoặc cập nhật 1 người trong DB (in-memory + file).
    """
    db[person_id] = {
        "name": name,
        "embedding": embedding,
        "registered_at": registered_at,
    }
    save_db(db)


def delete_person(db: dict, person_id: str) -> bool:
    """
    Xóa 1 người khỏi DB. Trả về True nếu xóa được, False nếu không tìm thấy.
    """
    if person_id not in db:
        return False
    del db[person_id]
    save_db(db)
    return True


def get_all_persons(db: dict) -> list[dict]:
    """
    Trả về danh sách người đã đăng ký (không kèm embedding vector).
    """
    return [
        {
            "person_id": pid,
            "name": record["name"],
            "registered_at": record["registered_at"],
        }
        for pid, record in db.items()
    ]


def build_embeddings_matrix(db: dict) -> tuple[Optional[np.ndarray], list[str]]:
    """
    Xây dựng ma trận embedding [N x 512] và danh sách person_id tương ứng.
    Dùng cho vectorized cosine distance — nhanh hơn vòng lặp Python khi N > 20.
    Trả về (None, []) nếu DB rỗng.
    """
    if not db:
        return None, []

    ids = list(db.keys())
    matrix = np.stack([db[pid]["embedding"] for pid in ids]).astype(np.float32)
    return matrix, ids
