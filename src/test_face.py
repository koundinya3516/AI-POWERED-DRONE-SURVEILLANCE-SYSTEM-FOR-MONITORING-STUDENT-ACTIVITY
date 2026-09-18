from deepface import DeepFace

result = DeepFace.find(
    img_path="test.jpeg",   # put your test image here
    db_path="data",
    enforce_detection=False
)

print(result)
