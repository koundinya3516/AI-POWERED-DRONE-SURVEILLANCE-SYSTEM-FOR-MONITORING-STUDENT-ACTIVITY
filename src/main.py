import cv2
import numpy as np
import urllib.request
from deepface import DeepFace
from collections import deque
import time
import os
import pickle

# ---------------- SETTINGS ----------------
ESP32_STREAM_URL = "http://10.202.245.18:81/stream"
DISPLAY_SIZE = (480, 320)

FACE_DIST_THRESHOLD = 0.9
CACHE_FILE = "face_db.pkl"

# --------- LOAD / CACHE DATABASE ----------
def load_database():
    if os.path.exists(CACHE_FILE):
        print("Loading cached database...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Creating face database...")
    database = {}

    for person in os.listdir("data"):
        person_path = os.path.join("data", person)

        if not os.path.isdir(person_path):
            continue

        embeddings = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            try:
                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name="VGG-Face",
                    enforce_detection=False
                )[0]["embedding"]

                rep = np.array(rep)
                norm = np.linalg.norm(rep)
                if norm != 0:
                    rep = rep / norm

                embeddings.append(rep)

            except:
                continue

        if embeddings:
            database[person] = np.mean(embeddings, axis=0)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(database, f)

    print("Database created & saved!")
    return database

database = load_database()
print("People loaded:", list(database.keys()))

# --------- STREAM ----------
def open_stream():
    while True:
        try:
            print("Connecting to ESP32...")
            stream = urllib.request.urlopen(ESP32_STREAM_URL, timeout=5)
            print("Connected!")
            return stream
        except:
            print("Retrying...")
            time.sleep(2)

stream = open_stream()
bytes_data = b''

# --------- ACTION BUFFERS ----------
frame_buffer = []
prediction_buffer = deque(maxlen=7)

motion_history = []
calibrated = False
idle_thr = None
agg_thr = None

print("Stay still for calibration...")

# --------- FUNCTIONS ----------
def get_motion(frames):
    motions = []
    for i in range(len(frames) - 1):
        diff = cv2.absdiff(frames[i], frames[i+1])
        motions.append(diff.sum())
    return float(np.mean(motions)) if motions else 0.0

def get_color(action):
    if action == "Normal":
        return (0, 255, 0)
    elif action == "Aggressive":
        return (0, 255, 255)
    else:
        return (0, 0, 255)

def recognize_face(face_img):
    try:
        rep = DeepFace.represent(
            img_path=face_img,
            model_name="VGG-Face",
            enforce_detection=False
        )[0]["embedding"]

        rep = np.array(rep)
        norm = np.linalg.norm(rep)
        if norm != 0:
            rep = rep / norm

        best_name = "Unknown"
        best_dist = 999

        for person, db_emb in database.items():
            dist = np.linalg.norm(rep - db_emb)

            if dist < best_dist:
                best_dist = dist
                best_name = person

        print("Match:", best_name, "Dist:", round(best_dist, 2))

        if best_dist < FACE_DIST_THRESHOLD:
            return best_name
        else:
            return "Unknown"

    except:
        return "Unknown"

# --------- MAIN LOOP ----------
while True:
    try:
        bytes_data += stream.read(1024)
    except:
        print("Stream error, reconnecting...")
        stream = open_stream()
        bytes_data = b''
        continue

    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')

    if a != -1 and b != -1 and b > a:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]

        if len(jpg) == 0:
            continue

        frame_array = np.frombuffer(jpg, dtype=np.uint8)
        if frame_array.size == 0:
            continue

        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        frame = cv2.resize(frame, DISPLAY_SIZE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --------- ACTION ---------
        frame_buffer.append(gray)
        if len(frame_buffer) > 5:
            frame_buffer.pop(0)

        action = "Calibrating..."

        if not calibrated and len(frame_buffer) == 5:
            motion = get_motion(frame_buffer)
            motion_history.append(motion)

            if len(motion_history) >= 100:
                base = np.mean(motion_history)
                idle_thr = base * 1.3
                agg_thr = base * 3
                calibrated = True
                print("Calibration done!")

        elif calibrated and len(frame_buffer) == 5:
            motion = get_motion(frame_buffer)

            if motion < idle_thr:
                label = "Normal"
            elif motion < agg_thr:
                label = "Aggressive"
            else:
                label = "Fight"

            prediction_buffer.append(label)
            action = max(set(prediction_buffer), key=prediction_buffer.count)

        # --------- FACE DETECTION ---------
        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False
            )
        except:
            faces = []

        for face in faces:
            try:
                fa = face["facial_area"]

                x = max(0, fa["x"])
                y = max(0, fa["y"])
                w = fa["w"]
                h = fa["h"]

                pad = 20
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)

                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                name = recognize_face(face_img)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, name,
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 0, 0), 2)

            except:
                continue

        # --------- DISPLAY ---------
        color = get_color(action)

        cv2.putText(frame, action,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        cv2.imshow("Drone Monitoring System", frame)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
