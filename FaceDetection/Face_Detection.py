import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model

facetracker = load_model("facetracker.h5")


faces_data = []

name = input("Enter your name: ")
id = input("Enter your id: ")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:

        y = int(sample_coords[1] * 450)
        x = int(sample_coords[0] * 450)
        y_add = int(sample_coords[3] * 450)
        x_add = int(sample_coords[2] * 450)
        cropped_image = frame[y : y_add, x : x_add, :]
        resized_image = cv2.resize(cropped_image, (100, 100))
        if len(faces_data) <= 300: 
            cv2.imwrite("dataset/User." + name + '_' + id + '.' + str(len(faces_data)) + ".jpg", resized_image)
            faces_data.append(resized_image)
        
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
            "face",
            tuple(
                np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -5])
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            str(len(faces_data)),
            (250, 250),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (255, 255, 255),
            1,
        )

    cv2.imshow("EyeTrack", frame)

    if (cv2.waitKey(1) & 0xFF == ord("q")) or len(faces_data) >= 300:
        break
cap.release()
cv2.destroyAllWindows()


