import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join("FaceDetection", "data", "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

number_images = 30

cap = cv2.VideoCapture(0)

for imgnum in range(number_images):
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH, f"{str(uuid.uuid1())}.jpg")
    cv2.imwrite(imgname, frame)
    cv2.putText(
        frame,
        str(imgnum + 1),
        (50, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 0, 0),
        1,
    )
    cv2.imshow("frame", frame)
    time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
