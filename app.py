from flask import Flask, request, render_template, redirect, url_for, session
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import hashlib
import json
import cv2
import time
import csv
import numpy as np
import tensorflow as tf
from keras.models import load_model
from connettion import mySql

from win32com.client import Dispatch

app = Flask(__name__)
app.secret_key = "Project AI"



today = datetime.today().strftime("%d_%m_%Y")

def read_id():
    ids = []
    with open('static/login.json', 'r') as file:
        data = json.load(file)
    for i in range(data):
        ids.append(data[i]["id"])
    return ids

def add_user(new_data):
    with open('static/login.json', 'r') as file:
        data = json.load(file)
    data.append(new_data)
    with open('static/login.json', 'w') as file:
        json.dump(data, file, indent=4)

def read_users():
    with open(os.path.join("static", "login.json")) as f:
        return json.load(f)

def validate_user(username, password):
    users = read_users()
    password = hashlib.md5(password.strip().encode("utf-8")).hexdigest()
    for user in users:
        if user["username"].strip() == username.strip() and user["password"] == password:
            return user
    return None

def extract_attendance():
    results = mySql.read(f"SELECT * FROM  {today}")
    return results

def mark_attendance(person):
    name = person.split('_')[0]
    id = person.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    exists = mySql.read(f"SELECT * FROM  {today} WHERE ids = {id}")
    if len(exists) == 0:
        try:
            mySql.insert(f"INSERT INTO {today} VALUE (%s, %s, %s)", (name, id, current_time))
        except Exception as e:
            print(e)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    err_msg = ""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = validate_user(username=username, password=password)
        if user:
            session["user"] = user
            return redirect(url_for("home"))
        else:
            err_msg = "Login failed"
    return render_template('login.html', err_msg=err_msg)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    err_msg = ""
    if request.method == "POST":
        new_data = {}
        new_data["id"] = request.form.get("id")
        new_data["name"] = request.form.get("name")
        new_data["username"] = request.form.get("username")
        new_data["password"] = request.form.get("password")
        con_password = request.form.get("password")
        ids = read_id()
        if new_data["id"] in ids:
            err_msg = "Account already exists"
            return render_template('signup.html', err_msg=err_msg)
        if new_data["password"] == con_password:
            new_data["password"] = hashlib.md5(new_data["password"].strip().encode("utf-8")).hexdigest()
            add_user(new_data=new_data)
            return redirect(url_for("home"))
        else:
            err_msg = "Passwords don't match"
    return render_template('signup.html', err_msg=err_msg)

@app.route('/logout')
def logout():
    session["user"] = None
    return redirect(url_for("home"))

@app.route('/attendance')
def attendance():
    mySql.create(f"CREATE TABLE IF NOT EXISTS {today} (names VARCHAR(30), ids INT, time VARCHAR(10))")
    userDetails = extract_attendance()
    print(userDetails)
    return render_template('attendance.html', l = len(userDetails), userDetails = userDetails ,today = today.replace("_", "-"))

@app.route('/start', methods=['GET'])
def start():
    if request.method == 'GET':
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
            _, frame = cap.read()
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
                mark_attendance(output[0])
                break
            if k == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for('attendance'))

@app.route('/add', methods=['POST'])
def add():
    err_msg = ""
    if request.method == "POST":
        faces_data = []
        names = []
        name = ""
        id = request.form.get("id")
        test = False
        with open('static/login.json', 'r') as file:
            data = json.load(file)
        for i in range(len(data)):
            if int(id) == data[i]["id"]:
                name = data[i]["name"]
                test = True
                break
        if not test:
            err_msg = "ID not initialized"
            return redirect(url_for("attendance", err_msg=err_msg))
        facetracker = load_model("facetracker.h5")
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
                    cv2.imwrite("dataset/User." + name + '_' + str(id) + '.' + str(len(faces_data)) + ".jpg", resized_image)
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
        faces = []
        names = []
        for image in os.listdir(os.path.join("dataset")):
            img = cv2.imread(os.path.join("dataset", image))
            name = image.split(".")[1]
            faces.append(img)
            names.append(name)

        faces = np.asarray(faces)
        faces = faces_data.reshape(len(faces), -1)

        with open("data/names.pkl", "wb") as f:
            pickle.dump(names, f)

        with open("data/faces_data.pkl", "wb") as f:
            pickle.dump(faces, f)
        return redirect(url_for("attendance", err_msg=err_msg))

if __name__ == '__main__':
    app.run(debug=True)
