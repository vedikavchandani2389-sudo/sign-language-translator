import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from collections import deque
from tensorflow.keras.models import load_model

# load your best model
model = load_model('best_model.keras')

# gestures same order as training
GESTURES = ['hello', 'yes', 'no','thanks', 'iloveyou', 'come', 'stop']

# setup text to speech
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# setup mediapipe
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

# function to extract landmarks
def extract_landmarks(result):
    if not result.hand_landmarks:
        return np.zeros(63)
    hand = result.hand_landmarks[0]
    landmarks = []
    for point in hand:
        landmarks.append(point.x)
        landmarks.append(point.y)
        landmarks.append(point.z)
    return np.array(landmarks)

# rolling window of 30 frames
sequence = deque(maxlen=30)

# variables to control speech
last_spoken = None
cooldown = 0
COOLDOWN_FRAMES = 45       # wait 45 frames before speaking again
CONFIDENCE_THRESHOLD = 0.85  # only speak if 85% confident

cap = cv2.VideoCapture(0)
print("Starting real time prediction...")
print("Show your hand and do a gesture!")
print("Press Q to quit")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # detect hand
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        result = landmarker.detect(mp_image)

        # draw dots on hand
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for point in hand:
                    h, w, _ = frame.shape
                    x = int(point.x * w)
                    y = int(point.y * h)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # extract landmarks and add to sequence
        landmarks = extract_landmarks(result)
        sequence.append(landmarks)

        # only predict when we have 30 frames
        if len(sequence) == 30:
            input_data = np.expand_dims(list(sequence), axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            confidence = np.max(prediction)
            gesture = GESTURES[np.argmax(prediction)]

            # show prediction on screen
            cv2.putText(frame, f'Gesture: {gesture}',
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0), 2)
            cv2.putText(frame, f'Confidence: {confidence*100:.0f}%',
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 0), 2)

            # speak only if confident enough and cooldown is done
            if confidence > CONFIDENCE_THRESHOLD and cooldown == 0:
                if gesture != last_spoken:
                    print(f"Speaking: {gesture} ({confidence*100:.0f}%)")
                    tts.say(gesture)
                    tts.runAndWait()
                    last_spoken = gesture
                    cooldown = COOLDOWN_FRAMES

        # reduce cooldown every frame
        if cooldown > 0:
            cooldown -= 1
            cv2.putText(frame, 'Speaking cooldown...',
                       (20, 150), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 0, 255), 2)

        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Closed successfully!")