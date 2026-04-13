"""
InsightFace wrapper — phát hiện, căn chỉnh, nhúng và so sánh khuôn mặt.

Pipeline:
  base64 image → decode → InsightFace (detect + align + ArcFace embed) → 512-dim vector
  → cosine distance với database → kết quả nhận diện
"""

import base64
import logging
from typing import Optional

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

from server.config import DET_SIZE, INSIGHTFACE_MODEL

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(self) -> None:
        """
        Khởi tạo InsightFace FaceAnalysis với model buffalo_l.
        ctx_id=0 tự động dùng GPU nếu có, ngược lại dùng CPU.
        """
        logger.info("Đang tải InsightFace model '%s'...", INSIGHTFACE_MODEL)
        self.analyzer = FaceAnalysis(name=INSIGHTFACE_MODEL)
        self.analyzer.prepare(ctx_id=0, det_size=DET_SIZE)
        logger.info("InsightFace model đã sẵn sàng.")

    # ------------------------------------------------------------------
    # Tiện ích xử lý ảnh
    # ------------------------------------------------------------------

    def decode_image(self, b64_string: str) -> Optional[np.ndarray]:
        """
        Giải mã base64 → numpy array BGR (định dạng OpenCV).
        Trả về None nếu dữ liệu không hợp lệ.
        """
        try:
            # Bỏ data URI prefix nếu có (vd: "data:image/jpeg;base64,...")
            if "," in b64_string:
                b64_string = b64_string.split(",", 1)[1]

            img_bytes = base64.b64decode(b64_string)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img  # BGR, shape (H, W, 3)
        except Exception as exc:
            logger.warning("Không decode được ảnh: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Phát hiện khuôn mặt lớn nhất trong ảnh, trả về embedding ArcFace 512-dim.
        Trả về None nếu không detect được khuôn mặt nào.

        InsightFace tự động:
          - Phát hiện vị trí khuôn mặt (RetinaFace)
          - Căn chỉnh theo 5 landmark (similarity transform)
          - Trích xuất ArcFace embedding đã L2-normalize
        """
        faces = self.analyzer.get(image)
        if not faces:
            return None

        # Chọn khuôn mặt lớn nhất (diện tích bbox) để tránh nhầm khuôn mặt phụ
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.embedding.astype(np.float32)

    def average_embeddings(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """
        Tính embedding trung bình từ nhiều ảnh, sau đó L2-normalize lại.
        Embedding trung bình từ nhiều góc chụp khác nhau giúp tăng độ robust.
        """
        stacked = np.stack(embeddings, axis=0)
        mean_emb = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        return mean_emb.astype(np.float32)

    # ------------------------------------------------------------------
    # So sánh / Nhận diện
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Tính cosine distance giữa 2 embedding.
        Vì InsightFace đã L2-normalize embedding, distance = 1 - dot(a, b).
        Giá trị: 0 = giống hệt, 2 = hoàn toàn khác. Thực tế: < 0.4 là cùng người.
        """
        return float(1.0 - np.dot(a, b))

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        embeddings_matrix: Optional[np.ndarray],
        person_ids: list[str],
        db: dict,
        threshold: float,
    ) -> tuple[Optional[str], Optional[str], float]:
        """
        Tìm người khớp nhất trong database bằng vectorized cosine distance.

        Args:
            query_embedding: embedding 512-dim của khuôn mặt cần nhận diện
            embeddings_matrix: ma trận [N x 512] của tất cả người trong DB
            person_ids: danh sách person_id tương ứng với từng hàng của matrix
            db: dict database (để lấy name)
            threshold: ngưỡng cosine distance

        Returns:
            (person_id, name, min_distance) nếu nhận ra
            (None, None, min_distance) nếu không nhận ra
        """
        if embeddings_matrix is None or len(person_ids) == 0:
            return None, None, 1.0

        # Vectorized: distances[i] = 1 - dot(query, matrix[i])
        distances = 1.0 - embeddings_matrix @ query_embedding
        min_idx = int(np.argmin(distances))
        min_distance = float(distances[min_idx])

        if min_distance <= threshold:
            pid = person_ids[min_idx]
            name = db[pid]["name"]
            return pid, name, min_distance

        return None, None, min_distance
