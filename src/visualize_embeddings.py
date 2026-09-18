import os
import numpy as np
import matplotlib.pyplot as plt
from deepface import DeepFace
from sklearn.decomposition import PCA

embeddings = []
labels = []

for person in os.listdir("data"):
    person_path = os.path.join("data", person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name="VGG-Face",
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append(rep)
            labels.append(person)

        except:
            continue

embeddings = np.array(embeddings)

# Reduce dimensions (important)
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(6,5))

for person in set(labels):
    idx = [i for i, l in enumerate(labels) if l == person]
    plt.scatter(reduced[idx, 0], reduced[idx, 1], label=person)

plt.legend()
plt.title("Face Embedding Visualization")
plt.savefig("embedding_plot.png")
plt.show()
