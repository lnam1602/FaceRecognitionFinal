"""
Cấu hình server — chỉnh sửa các giá trị này trước khi chạy trên Colab.
"""

# Đường dẫn file database khuôn mặt trên Google Drive
# Thay đổi nếu bạn dùng thư mục khác trên Drive
DRIVE_DB_PATH = "/content/drive/MyDrive/attendance/face_db.json"

# Fallback khi chạy local (không mount Drive)
LOCAL_DB_PATH = "data/face_db.json"

# Threshold cosine distance cho nhận diện
# Thấp hơn = chặt hơn. 0.4 phù hợp trong nhà, đủ sáng.
# Tăng lên 0.45-0.5 nếu dùng ngoài trời hoặc ánh sáng thay đổi nhiều.
DEFAULT_THRESHOLD = 0.4

# Kích thước ảnh đầu vào cho InsightFace detection
DET_SIZE = (640, 640)

# Tên model InsightFace (buffalo_l = RetinaFace detection + ArcFace recognition)
INSIGHTFACE_MODEL = "buffalo_l"

# API key để bảo vệ server — đặt vào header X-API-Key khi gọi API
# Thay bằng chuỗi ngẫu nhiên của bạn trước khi deploy
API_KEY = "change-me-before-deploy"

# Số lượng ảnh tối đa cho mỗi request đăng ký
MAX_REGISTER_IMAGES = 20
