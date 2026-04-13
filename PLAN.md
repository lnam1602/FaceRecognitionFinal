# Kế hoạch nâng cấp Face Recognition Server

## Context

Dự án nhận diện khuôn mặt cũ (4 năm) dùng Flask + VGG16 tự train + KNN trên raw pixel — độ chính xác thấp, chỉ chạy trên Windows, gắn chặt với HTML UI. Cần nâng cấp lên server API hiện đại chạy trên Google Colab để phục vụ app điểm danh với 2 tính năng: đăng ký khuôn mặt và nhận diện.

---

## Công nghệ được chọn

| Thành phần | Cũ | Mới | Lý do |
|---|---|---|---|
| Face Recognition | VGG16 + KNN (raw pixel 30,000 dims) | **InsightFace `buffalo_l`** (ArcFace, 512-dim embedding) | Accuracy tốt nhất, hỗ trợ khuôn mặt châu Á, tự động căn chỉnh khuôn mặt |
| API Framework | Flask + HTML templates | **FastAPI + Uvicorn** | REST API, async, tự sinh Swagger docs tại `/docs` |
| Tunnel Colab | N/A | **pyngrok** | 1 dòng code, không cần cấu hình thêm |
| Lưu trữ dữ liệu | MySQL + pickle file | **JSON trên Google Drive** | Colab không có MySQL; Drive persist qua các session |

---

## Cấu trúc file mới

```
face-recognition-server/
├── face_recognition_server.ipynb   ← Notebook chạy trên Colab
├── server/
│   ├── main.py          ← FastAPI app + routes + lifespan
│   ├── models.py        ← Pydantic schemas (request/response)
│   ├── face_service.py  ← InsightFace wrapper (detect, embed, compare)
│   ├── database.py      ← face_db.json read/write
│   └── config.py        ← Cấu hình (Drive path, threshold, ...)
├── requirements.txt
└── PLAN.md
```

**File cũ không còn dùng (có thể xóa):**
- `app.py`, `FaceRecognition.py`, `connettion.py` → thay bằng `server/`
- `facetracker.h5` (67 MB) → thay bằng InsightFace buffalo_l (auto-download)
- `templates/`, `static/` → không còn HTML UI
- `FaceDetection/` → InsightFace tự xử lý detection
- `data/*.pkl` → thay bằng `face_db.json` trên Google Drive

---

## API Endpoints

### POST /register — Đăng ký khuôn mặt

```
Request:
{
  "person_id": "SV001",
  "name": "Nguyen Van A",
  "images": ["base64JPEG...", "base64JPEG..."]   // 1-20 ảnh, khuyến nghị 3-10
}

Response 200:
{
  "status": "ok",
  "person_id": "SV001",
  "name": "Nguyen Van A",
  "embedding": [0.023, -0.417, ...],   // 512 floats — app lưu lại để cache offline
  "faces_detected": 5,
  "faces_failed": 1
}

Response 422 (không detect được mặt nào):
{ "status": "error", "detail": "No face detected in any of the provided images" }
```

### POST /recognize — Nhận diện khuôn mặt

```
Request:
{
  "image": "base64JPEG...",
  "threshold": 0.4    // optional, default 0.4. Thấp hơn = chặt hơn
}

Response — nhận ra:
{ "status": "recognized", "person_id": "SV001", "name": "Nguyen Van A", "confidence": 0.92, "distance": 0.18 }

Response — không nhận ra:
{ "status": "unknown", "confidence": null, "distance": 0.67 }

Response — không có mặt trong ảnh:
{ "status": "no_face" }
```

### GET /persons — Danh sách đã đăng ký

```
Response:
{
  "persons": [
    { "person_id": "SV001", "name": "Nguyen Van A", "registered_at": "2026-04-13T10:30:00" }
  ]
}
```

### DELETE /persons/{person_id} — Xóa người dùng

```
Response: { "status": "ok", "deleted": "SV001" }
```

### GET /health — Kiểm tra server

```
Response: { "status": "ok", "model_loaded": true, "registered_persons": 12 }
```

---

## Luồng dữ liệu

### Đăng ký
```
App → POST /register (ảnh base64)
  → InsightFace: detect + align + embed (512-dim) cho mỗi ảnh
  → average embeddings → 1 vector đại diện
  → ghi vào face_db.json trên Google Drive + cập nhật in-memory dict
  → trả về embedding cho app lưu local cache
```

### Nhận diện
```
App → POST /recognize (ảnh base64)
  → InsightFace: detect + align + embed query image
  → cosine distance với tất cả embeddings trong in-memory dict
  → min_distance < threshold → trả về person_id + name
```

**Schema face_db.json:**
```json
{
  "SV001": { "name": "Nguyen Van A", "embedding": [...512 floats...], "registered_at": "2026-04-13T10:30:00" }
}
```

> 30 sinh viên × 512 floats × 4 bytes ≈ **61 KB** — rất nhỏ, load/write tức thì.

---

## Cấu trúc Colab Notebook (5 cell)

| Cell | Nội dung | Chạy lại khi nào |
|---|---|---|
| 1 | Mount Google Drive | Mỗi session |
| 2 | `pip install` dependencies + tải InsightFace model | Lần đầu / sau reset runtime |
| 3 | `%%writefile` các file server | Khi thay đổi code |
| 4 | Khởi động uvicorn + ngrok → in ra public URL | Mỗi session |
| 5 | Quick test (health check) | Tùy chọn |

**Khi restart session:** Chỉ cần chạy Cell 1 và Cell 4. `face_db.json` trên Drive không bị mất.

---

## Bảo mật

API key đơn giản qua header `X-API-Key` — ngăn traffic ngẫu nhiên vào URL ngrok.  
Key được set trong `config.py` và chia sẻ với developer của app điểm danh.

---

## Lưu ý vận hành

- **Ngrok URL thay đổi mỗi session** — app điểm danh cần có màn hình settings để cập nhật URL
- **Colab free tier** tự disconnect sau 12h. Dùng Colab Pro cho ca học dài
- **Threshold mặc định 0.4** — phù hợp trong nhà, đủ sáng. Outdoor có thể tăng lên 0.45–0.5
- Không cần re-train: InsightFace `buffalo_l` là pretrained model (~300 MB, tự download)
- Model download lần đầu ~3 giây, các lần sau load từ cache (`~/.insightface/`)
