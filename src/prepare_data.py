import cv2
import numpy as np
import os

DATASET_PATH = "dataset"
SEQUENCE_LENGTH = 6

X = []
y = []

label_map = {
    "normal": 0,
    "aggressive": 1,
    "fight": 2
}

def extract_features(frames):
    motions = []
    flow_mags = []
    edge_changes = []

    for i in range(len(frames) - 1):
        f1 = frames[i]
        f2 = frames[i + 1]

        # -------- Frame Difference --------
        diff = cv2.absdiff(f1, f2)
        motions.append(diff.mean())

        # -------- Optical Flow --------
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(np.mean(mag))

        # -------- Edge Change --------
        e1 = cv2.Canny(f1, 50, 150)
        e2 = cv2.Canny(f2, 50, 150)
        edge_changes.append(np.mean(cv2.absdiff(e1, e2)))

    motions = np.array(motions)
    flow_mags = np.array(flow_mags)
    edge_changes = np.array(edge_changes)

    features = [
        np.mean(motions),
        np.std(motions),
        np.max(motions),

        np.mean(flow_mags),
        np.std(flow_mags),
        np.max(flow_mags),

        np.mean(edge_changes),
        np.std(edge_changes)
    ]

    return features


for label in label_map:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        cap = cv2.VideoCapture(path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

            if len(frames) == SEQUENCE_LENGTH:
                feat = extract_features(frames)
                X.append(feat)
                y.append(label_map[label])
                frames.pop(0)

        cap.release()

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Data prepared!")
print("X shape:", X.shape)
print("y shape:", y.shape)
