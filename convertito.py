import cv2
import os
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import VideoFileClip, AudioFileClip

relative_path = os.path.dirname(__file__)

colored_frames_output_folder = os.path.join(relative_path, 'tmp/colored_frames/')

out_path = os.path.join(relative_path, 'output/') 
out_video_name = 'colorized_video.mp4'
tmp_video_no_audio = os.path.join(out_path, 'tmp/no_audio_video.mp4')
out_video_full_path = os.path.join(out_path, out_video_name)
audio_path = os.path.join(relative_path, 'tmp/audio.wav')

frame_number = len(os.listdir(colored_frames_output_folder))

video = VideoFileClip('video.mp4')
duration_seconds = float(video.duration)    #durata file

#framerate = frame_number / duration_seconds    #numero file diviso durata video originale
framerate = video.fps
print(video.fps)
#print(framerate)

#converte frame in un video
cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame = cv2.imread(colored_frames_output_folder + '1.jpg')  
size = list(frame.shape)
del size[2]
size.reverse()

video_out = cv2.VideoWriter(out_video_full_path, cv2_fourcc, framerate, tuple(size))
for i in range(1, frame_number+1): 
    filename = colored_frames_output_folder + str(i) + '.jpg'
    if os.path.exists(filename):  
        frame = cv2.imread(filename)
        video_out.write(frame)
        print('frame %d' % i,' of', frame_number, end='\r')
        #print('frame ', i, ' of', frame_number, '')
    else:
        print(filename, ' not found')

video_out.release()

#applica audio al video

video = VideoFileClip(out_video_full_path)

audio = AudioFileClip(audio_path)

video = video.set_audio(audio)


video.write_videofile(out_video_full_path, audio_codec='aac')