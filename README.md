Title: Sign Language Gesture Recognizer

About:
I built this project to recognize basic sign language gestures in real time using your webcam. It detects your hand, reads the landmarks, and actually speaks the gesture out loud. Pretty cool to see it work live!

Gestures it can recognize:
hello · yes · no · thanks · i love you · come · stop

Tech used:
MediaPipe (hand tracking) + LSTM neural network + OpenCV + pyttsx3 (text to speech)

To run it yourself:
First install dependencies:
pip install tensorflow mediapipe opencv-python numpy scikit-learn pyttsx3
Then follow these 3 steps in order:
1. python collect_data.py` — show your hand gestures to the webcam, it'll record the data
2. python train_model.py` — trains the model on your data (grab a coffee, takes ~10 mins)
3. python realtime.py` — this is the fun part, it recognizes your gestures live and speaks them
