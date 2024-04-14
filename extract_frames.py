import cv2
import os

def extract_frames(video_path, output_folder, frame_interval):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = video.read()
        
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
            print(f"Extracted frame {frame_count}")

        frame_count += 1

    video.release()
    print(f"Finished extracting {extracted_count} frames.")

video_path = 'two.mp4'
output_folder = 'extracted_frames2'
frame_interval = 2

extract_frames(video_path, output_folder, frame_interval)
