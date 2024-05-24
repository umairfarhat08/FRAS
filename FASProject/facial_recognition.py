import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import threading
import time
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils

# Path to the directory containing images for attendance
path = 'imagesAttendance'

# Create the directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)

# Load images and their names from the directory
def load_images_from_directory(directory):
    images = []
    classNames = []
    try:
        imageList = os.listdir(directory)
        for imgName in imageList:
            imgPath = os.path.join(directory, imgName)
            currentImage = cv2.imread(imgPath)
            if currentImage is None:
                print(f"Error loading image {imgPath}")
                continue
            images.append(currentImage)
            classNames.append(os.path.splitext(imgName)[0])
    except Exception as e:
        print(f"Error loading images: {e}")
    return images, classNames

# Find face encodings
def findEncodings(images):
    encodeList = []
    for idx, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encodings = face_recognition.face_encodings(img)
            if encodings:
                encode = encodings[0]
                encodeList.append(encode)
                print(f"Encoding successful for image {idx + 1}")
            else:
                print(f"No face found in image {idx + 1}")
        except Exception as e:
            print(f"Error encoding image {idx + 1}: {e}")
    return encodeList

# Mark attendance in CSV
def markAttendance(name):
    try:
        if not os.path.isfile('Attendance.csv'):
            with open('Attendance.csv', 'w') as f:
                f.write('Name,Date,Time\n')

        with open('Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            namesList = [line.split(',')[0] for line in myDataList]

            if name not in namesList:
                now = datetime.now()
                date_string = now.strftime('%Y-%m-%d')
                time_string = now.strftime('%H:%M:%S')
                f.write(f'{name},{date_string},{time_string}\n')
                print(f"Attendance marked for {name} on {date_string} at {time_string}")
    except Exception as e:
        print(f"Error marking attendance: {e}")

# Calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Register a new face
def register_face(name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    count = 0
    start_time = time.time()
    success = False

    while count < 5 and (time.time() - start_time) < 10:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle
            face_img = frame[top:bottom, left:right]

            if time.time() - start_time > 2:
                img_path = f'{path}/{name}_{count}.jpg'
                cv2.imwrite(img_path, face_img)
                print(f"Image {count + 1} saved as {img_path}")
                count += 1
                start_time = time.time()

        cv2.imshow('Register Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 5:
        success = True
        messagebox.showinfo("Success", f"Registration successful for {name}")
    else:
        messagebox.showerror("Error", "Registration failed. Please try again.")

# Run facial recognition with liveness detection
def run_facial_recognition():
    images, classNames = load_images_from_directory(path)
    if len(images) == 0:
        print("No images loaded. Please register faces first.")
        return

    encodeListKnown = findEncodings(images)
    if len(encodeListKnown) == 0:
        print("No encodings found. Please check the images and try again.")
        return

    print('Encoding Complete')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    # Load dlib's face detector and create a predictor for facial landmarks
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(".venv/Scripts/shape_predictor_68_face_landmarks.dat")

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3
    blink_counter = 0
    blink_flag = False

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faces = detector(imgS, 0)
        facesCurrentFrame = face_recognition.face_locations(imgS)
        encodeCurrentFrame = face_recognition.face_encodings(imgS, facesCurrentFrame)

        for (i, face) in enumerate(faces):
            shape = predictor(imgS, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_flag = True
                blink_counter = 0

        for encodeFace, faceLoc in zip(encodeCurrentFrame, facesCurrentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDistances = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDistances)

            if matches[matchIndex] and blink_flag:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                markAttendance(name)
                blink_flag = False
            else:
                if blink_flag:
                    print("Blink detected but no matching face found.")
                else:
                    print("No blink detected.")
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to start facial recognition in a new thread
def start_facial_recognition():
    threading.Thread(target=run_facial_recognition).start()

# Function to prompt for a name and register a face
def register_new_face():
    name = simpledialog.askstring("Input", "Enter your name:")
    if name:
        threading.Thread(target=register_face, args=(name,)).start()

# Function to exit the application
def exit_application():
    app.destroy()

# Create the main application window
app = tk.Tk()
app.title("Facial Recognition Attendance System")

# Create a button to register a new face
register_button = tk.Button(app, text="Register New Face", command=register_new_face)
register_button.pack(pady=20)

# Create a button to start facial recognition
recognize_button = tk.Button(app, text="Start Facial Recognition", command=start_facial_recognition)
recognize_button.pack(pady=20)

# Create a button to exit the application
exit_button = tk.Button(app, text="Exit", command=exit_application)
exit_button.pack(pady=20)

# Run the application
app.mainloop()
