import cv2
import os
from moviepy.editor import VideoFileClip, AudioFileClip

def addAudio(out_video_full_path, audio_path):
    video = VideoFileClip(out_video_full_path)
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)
    video.write_videofile(out_video_full_path, audio_codec='aac')

def frameToVideo():
    # Define file paths
    relative_path = os.path.dirname(__file__)
    colored_frames_output_folder = os.path.join(relative_path, 'tmp', 'colored_frames')
    out_path = os.path.join(relative_path, 'output')
    out_video_name = 'colorized_video.mp4'
    out_video_full_path = os.path.join(out_path, out_video_name)
    audio_path = os.path.join(relative_path, 'tmp/audio.wav')

    # Create video writer object
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(os.path.join(colored_frames_output_folder, 'frame0.jpg'))
    size = frame.shape[:2][::-1]
    framerate = VideoFileClip('video.mp4').fps
    print(framerate)
    video_out = cv2.VideoWriter(out_video_full_path, cv2_fourcc, framerate, size)

    # Loop through colored frames and write to video file
    frame_number = len(os.listdir(colored_frames_output_folder))
    for i in range(frame_number):
        filename = os.path.join(colored_frames_output_folder, f'frame{i}.jpg')
        if os.path.exists(filename):
            frame = cv2.imread(filename)
            video_out.write(frame)
            print(f"Writing frame {i} of {frame_number}")
        else:
            print(f"File {filename} not found")

    # Release video writer to save file to disk
    video_out.release()
    addAudio(out_video_full_path, audio_path)