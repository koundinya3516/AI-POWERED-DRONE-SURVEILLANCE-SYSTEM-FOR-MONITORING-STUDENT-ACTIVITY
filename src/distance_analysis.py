import os
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace

same_dist = []
diff_dist = []

database = {}

# -------- LOAD EMBEDDINGS --------
for person in os.listdir("data"):
    person_path = os.path.join("data", person)

    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name="VGG-Face",
                enforce_detection=False
            )[0]["embedding"]

            rep = np.array(rep)
            rep = rep / np.linalg.norm(rep)

            embeddings.append(rep)

        except:
            continue

    database[person] = embeddings

# -------- SAME PERSON DISTANCES --------
for person, emb_list in database.items():
    for i in range(len(emb_list)):
        for j in range(i + 1, len(emb_list)):
            d = np.linalg.norm(emb_list[i] - emb_list[j])
            same_dist.append(d)

# -------- DIFFERENT PERSON DISTANCES --------
people = list(database.keys())

for i in range(len(people)):
    for j in range(i + 1, len(people)):
        for e1 in database[people[i]]:
            for e2 in database[people[j]]:
                d = np.linalg.norm(e1 - e2)
                diff_dist.append(d)

# -------- PLOT --------
plt.figure()
plt.hist(same_dist, bins=20, alpha=0.7, label="Same Person")
plt.hist(diff_dist, bins=20, alpha=0.7, label="Different Person")

plt.legend()
plt.title("Face Distance Distribution")
plt.xlabel("Distance")
plt.ylabel("Frequency")

plt.savefig("distance_plot.png")
plt.show()

print("Distance plot saved as distance_plot.png")
