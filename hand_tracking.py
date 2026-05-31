import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

def extract_landmarks(result):
    # if no hand detected, return 63 zeros
    if not result.hand_landmarks:
        return np.zeros(63)
    
    # get first hand's landmarks
    hand = result.hand_landmarks[0]
    
    # extract x, y, z from all 21 points
    landmarks = []
    for point in hand:
        landmarks.append(point.x)
        landmarks.append(point.y)
        landmarks.append(point.z)
    
    # returns array of 63 numbers
    return np.array(landmarks)

cap = cv2.VideoCapture(0)
print("Show your hand! Press Q to quit")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        result = landmarker.detect(mp_image)

        # extract the 63 numbers
        landmarks = extract_landmarks(result)

        # draw dots on hand
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for point in hand:
                    h, w, _ = frame.shape
                    x = int(point.x * w)
                    y = int(point.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # print first 9 numbers so you can see them
            print(f"Landmarks: {landmarks[:9].round(3)}")

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()