import cv2
import mediapipe as mp
import numpy as np
import os
import time

# all your gestures
GESTURES = ['hello', 'yes', 'no', 'thanks', 'iloveyou', 'come', 'stop']

# how much data to collect
SEQUENCES = 30   # 30 videos per gesture
FRAMES = 30      # 30 frames per video

# folder where data will be saved
DATA_PATH = 'dataset'

# mediapipe setup
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

# function to extract 63 numbers from hand
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

# create folders for each gesture automatically
for gesture in GESTURES:
    for seq in range(SEQUENCES):
        folder = os.path.join(DATA_PATH, gesture, str(seq))
        os.makedirs(folder, exist_ok=True)

print("Folders created successfully!")

# start camera and collect data
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    
    # loop through each gesture
    for gesture in GESTURES:
        print(f"\n--- GET READY FOR: {gesture.upper()} ---")
        
        # loop through each sequence
        for seq in range(SEQUENCES):
            
            # countdown before each sequence
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                
                # show gesture name and countdown on screen
                cv2.putText(frame, f'Gesture: {gesture.upper()}', 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.putText(frame, f'Sequence: {seq+1}/{SEQUENCES}', 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 0), 2)
                cv2.putText(frame, f'Starting in {countdown}...', 
                           (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 0, 255), 2)
                
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1000)  # wait 1 second
            
            # collect 30 frames for this sequence
            for frame_num in range(FRAMES):
                ret, frame = cap.read()
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
                
                # show recording status on screen
                cv2.putText(frame, f'Gesture: {gesture.upper()}', 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.putText(frame, f'Sequence: {seq+1}/{SEQUENCES}', 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 0), 2)
                cv2.putText(frame, f'Frame: {frame_num+1}/{FRAMES}', 
                           (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 0, 0), 2)
                cv2.putText(frame, 'RECORDING...', 
                           (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 0, 255), 2)
                
                cv2.imshow("Data Collection", frame)
                cv2.waitKey(1)
                
                # save landmarks to file
                landmarks = extract_landmarks(result)
                save_path = os.path.join(DATA_PATH, gesture, str(seq), str(frame_num))
                np.save(save_path, landmarks)
            
            print(f"Sequence {seq+1}/{SEQUENCES} done for {gesture}")
        
        print(f"\n{gesture.upper()} COMPLETE!")
        print("Rest for a few seconds before next gesture...")
        time.sleep(3)

cap.release()
cv2.destroyAllWindows()
print("\nAll data collected successfully!")