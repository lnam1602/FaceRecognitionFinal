import os
import cv2
import pickle
import numpy as np

faces_data = []
names = []
for image in os.listdir(os.path.join("dataset")):
    img = cv2.imread(os.path.join("dataset", image))
    name = image.split(".")[1]
    faces_data.append(img)
    names.append(name)

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(len(faces_data), -1)

with open("data/names.pkl", "wb") as f:
    pickle.dump(names, f)

with open("data/faces_data.pkl", "wb") as f:
    pickle.dump(faces_data, f)