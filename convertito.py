import cv2
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip, AudioFileClip

dirname = os.path.dirname(__file__)
print(dirname)
path = 'framec/'   
files = os.listdir(path)
num_files = len(files)  #numero file

video = VideoFileClip('video.mp4')
duration_seconds = float(video.duration)    #durata file

framerate = num_files / duration_seconds    #numero file diviso durata video originale

#converte frame in un video
out_path = dirname
out_video_name = 'convertito.mp4'
out_video_full_path = os.path.join(out_path, out_video_name)

cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame = cv2.imread(path + '1.jpg')  
size = list(frame.shape)
del size[2]
size.reverse()

video_out = cv2.VideoWriter(out_video_full_path, cv2_fourcc, framerate, tuple(size))
for i in range(1, num_files+1): 
    filename = path + str(i) + '.jpg'
    if os.path.exists(filename):  
        frame = cv2.imread(filename)
        video_out.write(frame)
        print('frame ', i, ' of', num_files, '')
    else:
        print(filename, ' not found')

video_out.release()
print('outputed video to ', out_video_full_path)

#applica audio al video
audio_path = os.path.join(dirname, 'audio.wav')

video_path = os.path.join(dirname, 'convertito.mp4')

video = VideoFileClip(out_video_full_path)

audio = AudioFileClip(audio_path)

video = video.set_audio(audio)


video.write_videofile('convertito_con_audio.mp4', audio_codec='aac')