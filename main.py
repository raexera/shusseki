import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

path = "training"
images = list()
person_names = list()
person_list = os.listdir(path)

for cu_image in person_list:
    if cu_image.endswith((".jpg", ".jpeg", ".png")):
        current_image = cv2.imread(os.path.join(path, cu_image))
        if current_image is None:
            print(f"Error: Unable to load image '{cu_image}'")
        else:
            images.append(current_image)
            person_names.append(os.path.splitext(cu_image)[0])
    else:
        print(f"Skipping non-image file: '{cu_image}'")

print(person_list)
print(person_names)

def face_encodings(_images):
    encode_list = []
    for image in _images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image_rgb)
        if len(encode) > 0:
            encode_list.append(encode[0])
        else:
            print("Error: No face found in the image")
    return encode_list


def attendance(_name):
    if not os.path.exists("Attendance.csv"):
        with open("Attendance.csv", "w") as f:
            f.write("Name,Time,Date\n")

    with open("Attendance.csv", "a") as f:
        time_now = datetime.now()
        time_string = time_now.strftime("%H:%M:%S")
        date_string = time_now.strftime("%d/%m/%Y")
        f.write(f"{_name},{time_string},{date_string}\n")


THRESHOLD = 0.5

encode_list_known = face_encodings(images)
print("Encodings completed.")

cap = cv2.VideoCapture(0)
correct_recognitions = 0
total_recognitions = 0
face_detected = False
countdown_started = False
start_time = 0

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    faces_current_frame = face_recognition.face_locations(faces)
    encodes_current_frame = face_recognition.face_encodings(faces, faces_current_frame)

    for encode_face, faceLoc in zip(encodes_current_frame, faces_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        matchIndex = np.argmin(face_dis)

        if matches[matchIndex] and face_dis[matchIndex] < THRESHOLD:
            correct_recognitions += 1
        total_recognitions += 1

        if matches[matchIndex]:
            name = person_names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 215), 3)
            cv2.rectangle(
                frame, (x1, y2 - 35), (x2, y2), (0, 255, 215), cv2.RETR_FLOODFILL
            )
            cv2.putText(
                frame,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            if not face_detected:
                face_detected = True
                countdown_started = True
                start_time = time.time()

            if countdown_started and time.time() - start_time >= 10:
                attendance(name)
                countdown_started = False
                face_detected = False

        else:
            if face_detected:
                face_detected = False
                countdown_started = False

    cv2.imshow("Automated Attendance | Camera", frame)

    keys = cv2.waitKey(1) & 0xFF
    if keys == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

accuracy = correct_recognitions / total_recognitions * 100
print(f"Accuracy: {accuracy}%")