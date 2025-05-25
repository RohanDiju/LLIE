import cv2
import numpy as np
import tensorflow as tf
import threading
from queue import Queue
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D
import tensorflow.keras.backend as K

# Load pre-trained Zero-DCE model
def load_model(size):
    input_img = Input(shape=(size[1], size[0], 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)

    int_con1 = Concatenate(axis=-1)([conv4, conv3])
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(int_con1)
    int_con2 = Concatenate(axis=-1)([conv5, conv2])
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(int_con2)
    int_con3 = Concatenate(axis=-1)([conv6, conv1])
    x_r = Conv2D(24, (3,3), activation='tanh', padding='same')(int_con3)

    model = Model(inputs=input_img, outputs=x_r)
    model.load_weights("weights/best.h5")
    return model

# Initialize model
model = load_model((640, 480))  # Adjust based on video resolution

def enhance_frame(frame):
    """Enhance a single frame using Zero-DCE model."""
    frame_array = np.asarray(frame) / 255.0
    input_data = np.expand_dims(frame_array, 0)
    A = model.predict(input_data)
    
    # Compute enhancement
    r_layers = [A[:, :, :, i:i+3] for i in range(0, 24, 3)]
    x = frame_array
    for r in r_layers:
        x = x + r * (K.pow(x, 2) - x)

    enhanced_image = (x[0].numpy() * 255).astype(np.uint8)
    return enhanced_image

# Queue to store processed frames
frame_queue = Queue()

def process_frames():
    """Process frames asynchronously."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            enhanced = enhance_frame(frame)
            cv2.imshow("Enhanced Video", enhanced)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video path

# Start processing thread
threading.Thread(target=process_frames, daemon=True).start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    
    # Add frame to queue for processing
    frame_queue.put(frame)

cap.release()
cv2.destroyAllWindows()
