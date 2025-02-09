import os
import cv2
import numpy as np
import tarfile
from PIL import Image
from io import BytesIO

upload_folder = 'inputs'
result_folder = 'results'
frames_folder = 'frames'
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)

IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv')

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    frames_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_folder, f'frame_{frame_count:05d}.png')
        cv2.imwrite(frame_path, frame)
        frames_list.append(frame_path)
        frame_count += 1
    
    cap.release()
    return frames_list, frame_rate

def reconstruct_video(frames_list, output_video_path, frame_rate):
    first_frame = cv2.imread(frames_list[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    for frame_path in frames_list:
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f'Video saved at {output_video_path}')

def process_video(video_path):
    frames_list, frame_rate = extract_frames(video_path)
    upscaled_frames = []
    
    for frame_path in frames_list:
        image = Image.open(frame_path).convert('RGB')
        sr_image = model.predict(np.array(image))
        sr_frame_path = frame_path.replace('frames/', 'results/')
        sr_image.save(sr_frame_path)
        upscaled_frames.append(sr_frame_path)
    
    output_video_path = os.path.join(result_folder, os.path.basename(video_path))
    reconstruct_video(upscaled_frames, output_video_path, frame_rate)

def process_input(filename):
    if filename.endswith(VIDEO_FORMATS):
        process_video(filename)
    elif tarfile.is_tarfile(filename):
        process_tar(filename)
    else:
        result_image_path = os.path.join(result_folder, os.path.basename(filename))
        image = Image.open(filename).convert('RGB')
        sr_image = model.predict(np.array(image))
        sr_image.save(result_image_path)
        print(f'Finished! Image saved to {result_image_path}')

uploaded = "sample_data1.mp4"
print('Processing:', uploaded)
process_input(uploaded)
