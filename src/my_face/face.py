from deepface import DeepFace

def recognize_face(frame):
    try:
        result = DeepFace.find(
            img_path=frame,
            db_path="data",
            enforce_detection=False,
            detector_backend="opencv"
        )

        if len(result) > 0 and len(result[0]) > 0:
            return result[0].iloc[0]['identity'].split('/')[-2]

    except Exception as e:
        print("Face error:", e)

    return "Unknown"
