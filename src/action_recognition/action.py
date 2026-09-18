import cv2

prev_frame = None

def detect_action(frame):
    global prev_frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    action = "Normal"

    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, gray)
        motion = diff.sum()

        if motion > 1000000:
            action = "Aggressive"

    prev_frame = gray
    return action
