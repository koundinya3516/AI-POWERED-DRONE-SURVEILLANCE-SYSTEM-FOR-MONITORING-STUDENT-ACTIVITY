import cv2
import os
import pickle

data_path = "data"
known_faces = []
known_names = []

def get_feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (100, 100)).flatten()

for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)

    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        feature = get_feature(img)
        known_faces.append(feature)
        known_names.append(person)

with open("models.pkl", "wb") as f:
    pickle.dump((known_faces, known_names), f)

print("Encodings saved ✔")
