import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt

path = "training"
images = []
person_names = []
person_nims = []
person_list = os.listdir(path)
attendance_dict = {}
ATTENDANCE_TIME = 1 * 60  # 100 minutes
MINIMUM_REQUIRED_TIME = 0.1 * ATTENDANCE_TIME  # 80% of 100 minutes
results = []
recorded_nims = set()  # To keep track of recorded NIMs

for cu_image in person_list:
    if cu_image.endswith((".jpg", ".jpeg", ".png")):
        current_image = cv2.imread(os.path.join(path, cu_image))
        if current_image is None:
            print(f"Error: Unable to load image '{cu_image}'")
        else:
            images.append(current_image)
            # Extract NIM and name from the filename
            filename = os.path.splitext(cu_image)[0]
            if '-' in filename:
                nim, name = filename.split('-')
            elif '_' in filename:
                nim, name = filename.split('_')
            else:
                print(f"Filename '{filename}' does not match expected format. Skipping.")
                continue
            person_nims.append(nim.strip())
            person_names.append(name.strip())
    else:
        print(f"Skipping non-image file: '{cu_image}'")

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

def save_attendance(nim, name):
    if nim in recorded_nims:
        return

    time_now = datetime.now()
    time_string = time_now.strftime("%H:%M:%S")
    date_string = time_now.strftime("%d/%m/%Y")
    status = "P"  # Present

    # Save to CSV
    if not os.path.exists("Attendance.csv"):
        with open("Attendance.csv", "w") as f:
            f.write("NIM,NAME,DATE,TIME,STATUS\n")

    with open("Attendance.csv", "a") as f:
        f.write(f"{nim},{name},{date_string},{time_string},{status}\n")

    recorded_nims.add(nim)  # Mark this NIM as recorded

THRESHOLD = 0.5

encode_list_known = face_encodings(images)
print("Encodings completed.")

cap = cv2.VideoCapture(0)
correct_recognitions = 0
total_recognitions = 0

start_time = time.time()
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
            nim = person_nims[matchIndex].upper()
            name = person_names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 215), 3)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 215), cv2.RETR_FLOODFILL)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if nim not in attendance_dict:
                attendance_dict[nim] = {
                    "name": name,
                    "start_time": time.time(),
                    "total_time": 0,
                    "is_present": True
                }
            else:
                if not attendance_dict[nim]["is_present"]:
                    attendance_dict[nim]["start_time"] = time.time()
                    attendance_dict[nim]["is_present"] = True

            attendance_dict[nim]["total_time"] = time.time() - attendance_dict[nim]["start_time"]

            if attendance_dict[nim]["total_time"] >= MINIMUM_REQUIRED_TIME:
                if attendance_dict[nim]["total_time"] < ATTENDANCE_TIME:
                    save_attendance(nim, name)
                attendance_dict[nim]["total_time"] = 0
        else:
            for nim in attendance_dict:
                if attendance_dict[nim]["is_present"]:
                    attendance_dict[nim]["is_present"] = False

    elapsed_time = time.time() - start_time
    results.append((elapsed_time, correct_recognitions, total_recognitions))

    cv2.imshow("Automated Attendance | Camera", frame)

    keys = cv2.waitKey(1) & 0xFF
    if keys == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Save results to CSV
df = pd.DataFrame(results, columns=['Elapsed Time', 'Correct Recognitions', 'Total Recognitions'])
df.to_csv('recognition_results.csv', index=False)

accuracy = correct_recognitions / total_recognitions * 100
print(f"Accuracy: {accuracy}%")

# Plot the results
times = [result[0] for result in results]
accuracies = [(result[1] / result[2]) * 100 if result[2] > 0 else 0 for result in results]

plt.plot(times, accuracies, label='Accuracy')
plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Accuracy (%)')
plt.title('Recognition Accuracy Over Time')
plt.legend()
plt.show()
