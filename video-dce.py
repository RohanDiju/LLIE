import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
from moviepy.editor import VideoFileClip
from PIL import Image
from src.model import DCE_x
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D
import tensorflow.keras.backend as K

# Argument parser
parser = argparse.ArgumentParser(description="Zero-DCE video enhancement with frame skipping and interpolation.")
parser.add_argument('--input_video', type=str, required=True, help='Input video file.')
parser.add_argument('--output_video', type=str, default="output.mkv", help='Output video file.')
parser.add_argument('--max_frames', type=int, default=None, help="Maximum frames to process (useful for testing).")
parser.add_argument('--input_fps', type=int, default=15, help="Reduce input FPS before processing.")
parser.add_argument('--output_fps', type=int, default=15, help="FPS of the output video.")
parser.add_argument('--frame_skip', type=int, default=2, help="Process every Nth frame and interpolate the rest.")
args = parser.parse_args()

def load_model(size):
    """Load the Zero-DCE model with pre-trained weights."""
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

def enhance_frame(frame, model):
    """Enhance a single frame using Zero-DCE model."""
    frame_array = np.asarray(frame) / 255.0
    input_data = np.expand_dims(frame_array, 0)
    A = model.predict(input_data)
    
    r_values = [A[:, :, :, i:i+3] for i in range(0, 24, 3)]
    
    x = frame_array
    for r in r_values:
        x = x + r * (K.pow(x, 2) - x)

    enhanced_image = (x[0].numpy() * 255).astype(np.uint8)
    return enhanced_image

def interpolate_frames(frame1, frame2, alpha):
    """Linearly interpolate between two frames."""
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

def process_video(input_video, output_video, max_frames, input_fps, output_fps, frame_skip):
    """Process video with frame skipping and interpolation."""
    video = VideoFileClip(input_video).subclip(0, max_frames / video.fps if max_frames else None)
    video = video.set_fps(input_fps)  # Reduce FPS at input level
    model = load_model(video.size)

    processed_frames = []
    frame_buffer = []
    total_frames = int(video.fps * video.duration)
    
    print(f"Processing {total_frames} frames (Skipping every {frame_skip-1} frames)...")

    for i, frame in enumerate(video.iter_frames(fps=input_fps)):
        if i % frame_skip == 0:
            enhanced_frame = enhance_frame(frame, model)
            frame_buffer.append((i, enhanced_frame))
            processed_frames.append(enhanced_frame)

    print(f"Frames processed: {len(processed_frames)}. Interpolating missing frames...")

    # Interpolation for skipped frames
    full_video = []
    for j in range(len(frame_buffer) - 1):
        start_idx, frame_start = frame_buffer[j]
        end_idx, frame_end = frame_buffer[j + 1]

        full_video.append(frame_start)
        for k in range(1, frame_skip):
            alpha = k / frame_skip
            interpolated_frame = interpolate_frames(frame_start, frame_end, alpha)
            full_video.append(interpolated_frame)

    full_video.append(frame_buffer[-1][1])  # Add last frame

    print(f"Final video has {len(full_video)} frames.")

    # Convert processed frames to video using OpenCV
    height, width, _ = full_video[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, output_fps, (width, height))

    for frame in full_video:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"Processed video saved as {output_video}")

def main():
    process_video(
        args.input_video, args.output_video, args.max_frames, 
        args.input_fps, args.output_fps, args.frame_skip
    )

if __name__ == "__main__":
    main()
