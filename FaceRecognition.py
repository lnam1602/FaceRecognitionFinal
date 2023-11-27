from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
import tensorflow as tf
from datetime import datetime
from keras.models import load_model

from win32com.client import Dispatch


def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)


facetracker = load_model("facetracker.h5")

with open("data/names.pkl", "rb") as w:
    LABELS = pickle.load(w)
with open("data/faces_data.pkl", "rb") as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

COL_NAMES = ["NAME", "TIME"]

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[50:500, 50:500, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        # Add face_images
        y = int(sample_coords[1] * 450)
        x = int(sample_coords[0] * 450)
        y_add = int(sample_coords[3] * 450)
        x_add = int(sample_coords[2] * 450)
        cropped_image = frame[y : y_add, x : x_add, :]
        resized_image = cv2.resize(cropped_image, (100, 100)).flatten().reshape(1, -1)
        output = knn.predict(resized_image)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        # Controls the main rectangle
        cv2.rectangle(
            frame,
            tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
            tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
            (255, 0, 0),
            2,
        )

        # Controls the label rectangle
        cv2.rectangle(
            frame,
            tuple(
                np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -30])
            ),
            tuple(
                np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [80, 0])
            ),
            (255, 0, 0),
            -1,
        )

        # Controls the text rendered
        cv2.putText(
            frame,
            str(output[0].split("_")[0].split(" ")[len(output[0].split("_")[0].split(" ")) - 1]),
            tuple(
                np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -10])
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        attendance = [str(output[0]), str(timestamp)]
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord("o"):
        speak("Attendance Success")
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
        break
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
