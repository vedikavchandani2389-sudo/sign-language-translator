import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# your gestures - same order as before
GESTURES = ['hello', 'yes', 'no', 'thanks', 'iloveyou', 'come', 'stop']

DATA_PATH = 'dataset'
SEQUENCES = 30
FRAMES = 30

print("Loading your data...")

# load all saved files into arrays
X = []  # landmark data
y = []  # gesture labels

for label_idx, gesture in enumerate(GESTURES):
    for seq in range(SEQUENCES):
        # collect all 30 frames for this sequence
        sequence_frames = []
        for frame_num in range(FRAMES):
            path = os.path.join(DATA_PATH, gesture, str(seq), str(frame_num) + '.npy')
            frame_data = np.load(path)
            sequence_frames.append(frame_data)
        
        X.append(sequence_frames)
        y.append(label_idx)
    
    print(f"Loaded {gesture} data!")

# convert to numpy arrays
X = np.array(X)   # shape: (210, 30, 63)
y = np.array(y)   # shape: (210,)

print(f"\nTotal sequences: {X.shape[0]}")
print(f"Frames per sequence: {X.shape[1]}")
print(f"Landmarks per frame: {X.shape[2]}")

# one hot encode labels
# converts 0,1,2,3,4,5,6 into arrays like [1,0,0,0,0,0,0]
y_categorical = to_categorical(y, num_classes=len(GESTURES))

# split into training and testing
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, 
    test_size=0.2, 
    random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# build the LSTM model
print("\nBuilding model...")

model = Sequential([
    # first LSTM layer
    # return_sequences=True means pass full sequence to next layer
    LSTM(64, return_sequences=True, input_shape=(FRAMES, 63)),
    Dropout(0.2),  # randomly turn off 20% neurons to prevent overfitting
    
    # second LSTM layer
    # return_sequences=False means only pass final output
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    
    # dense layers for final classification
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    # output layer - one neuron per gesture
    Dense(len(GESTURES), activation='softmax')
])

# compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# print model summary
model.summary()

# save best model automatically during training
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# train the model!
print("\nStarting training... this will take 5-15 minutes")
print("Watch the accuracy go up each epoch!\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

# evaluate on test data
print("\nEvaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {accuracy*100:.1f}%")

# save final model
model.save('gesture_model.keras')
print("\nModel saved as gesture_model.keras!")
print("Best model saved as best_model.keras!")
print("\nTraining complete!")