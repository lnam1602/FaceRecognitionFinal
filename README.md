# Face Recognition Server

Server nhận diện khuôn mặt chạy trên **Google Colab**, phục vụ app điểm danh qua REST API.

## Tính năng

- **Đăng ký khuôn mặt** — gửi ảnh, nhận về embedding để app lưu local
- **Nhận diện khuôn mặt** — gửi ảnh, nhận về tên/mã người dùng
- **Swagger UI** tự động tại `/docs`
- Dữ liệu lưu trên **Google Drive**, không mất khi Colab reset

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Face Recognition | InsightFace `buffalo_l` (ArcFace, 512-dim embedding) |
| API Framework | FastAPI + Uvicorn |
| Tunnel | pyngrok (ngrok) |
| Lưu trữ | JSON trên Google Drive |

## Cấu trúc project

```
face-recognition-server/
├── face_recognition_server.ipynb   # Notebook chạy trên Colab
├── server/
│   ├── main.py          # FastAPI app + 5 endpoints
│   ├── models.py        # Request/response schemas
│   ├── face_service.py  # InsightFace wrapper
│   ├── database.py      # Đọc/ghi face_db.json
│   └── config.py        # Cấu hình (API key, threshold,...)
├── data/                # face_db.json lưu ở đây khi không dùng Drive
├── requirements.txt
└── PLAN.md              # Chi tiết kế hoạch nâng cấp
```

## Hướng dẫn chạy trên Google Colab

### Bước 1 — Chuẩn bị

1. Lấy **ngrok auth token** miễn phí tại [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)
2. Mở [server/config.py](server/config.py) trong notebook và đổi:
   - `API_KEY` thành chuỗi bí mật của bạn
   - `NGROK_AUTH_TOKEN` trong Cell 4

### Bước 2 — Chạy lần đầu

Chạy **tuần tự** 5 cell trong `face_recognition_server.ipynb`:

| Cell | Làm gì |
|------|--------|
| 1 | Mount Google Drive |
| 2 | Cài dependencies + tải InsightFace model (~300 MB, 1 lần) |
| 3 | Ghi code server vào Colab |
| 4 | Khởi động uvicorn + ngrok — **copy URL output** |
| 5 | Kiểm tra nhanh (tùy chọn) |

### Bước 3 — Các lần sau (restart session)

Chỉ cần chạy **Cell 1** và **Cell 4**. Dữ liệu trên Drive không bị mất.

> **Lưu ý:** Ngrok URL thay đổi mỗi session. Nhớ cập nhật URL trong app điểm danh.

## API Reference

Tất cả request dùng JSON. Ảnh gửi dưới dạng **base64** trong body.
Các endpoint (trừ `/health`) yêu cầu header: `X-API-Key: <your-key>`

### POST `/register` — Đăng ký khuôn mặt

```json
// Request
{
  "person_id": "SV001",
  "name": "Nguyen Van A",
  "images": ["<base64>", "<base64>"]
}

// Response
{
  "status": "ok",
  "person_id": "SV001",
  "name": "Nguyen Van A",
  "embedding": [0.023, ...],
  "faces_detected": 5,
  "faces_failed": 1
}
```

### POST `/recognize` — Nhận diện khuôn mặt

```json
// Request
{ "image": "<base64>", "threshold": 0.4 }

// Nhận ra
{ "status": "recognized", "person_id": "SV001", "name": "Nguyen Van A", "confidence": 0.92, "distance": 0.18 }

// Không nhận ra
{ "status": "unknown", "distance": 0.67 }

// Không có mặt trong ảnh
{ "status": "no_face" }
```

### GET `/persons` — Danh sách đã đăng ký

### DELETE `/persons/{person_id}` — Xóa người dùng

### GET `/health` — Kiểm tra server (không cần API key)

## Chạy local (tùy chọn)

```bash
pip install -r requirements.txt
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập Swagger UI tại `http://localhost:8000/docs`

> Khi chạy local, `face_db.json` được lưu tại `data/face_db.json` thay vì Google Drive.

## Lưu ý vận hành

- **Threshold mặc định 0.4** — phù hợp trong nhà, đủ sáng. Tăng lên 0.45–0.5 nếu dùng ngoài trời
- **Colab free tier** tự disconnect sau 12h. Dùng Colab Pro cho ca học dài hơn
- Khuyến nghị gửi **3–10 ảnh** khi đăng ký để tăng độ chính xác
