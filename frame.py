from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import cv2
import os
from cv2 import dnn
import sys
from PIL import Image

dirname = os.path.dirname(__file__)

#prende l'audio dal video e lo converte in wav
video_path = os.path.join(dirname, 'video.mp4')
audio_path = os.path.join(dirname, 'audio.wav')


if os.path.exists(video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)
    audio_clip.close()
    video_clip.close()

#stabilisce i vari path
save_dir = os.path.join(dirname, 'frames')
save_dir_color = os.path.join(dirname, 'framec')     #cartella da sistemare
#applica i tre modelli
proto_file = os.path.join(dirname, 'colorization_deploy_v2.prototxt')
model_file = os.path.join(dirname, 'colorization_release_v2.caffemodel')
hull_pts = os.path.join(dirname, 'pts_in_hull.npy')

#converte il video in bianco e nero in frame
if not os.path.exists(save_dir):
    os.makedirs(save_dir) and os.makedirs(save_dir_color)       #sistemare non crea framec

vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 0

while success:
    cv2.imwrite(os.path.join(save_dir, f"frame{count}.jpg"), image)     
    success, image = vidcap.read()
    count += 1
    print(count)

    if cv2.waitKey(10) == 27:                
        break

vidcap.release()
frame = cv2.imread(os.path.join(save_dir, "frame0.jpg"))
height, width, _ = frame.shape

count -= 1
print ("frames " + str(count))   #quantità file presenti nella cartella dei frame

#converte i video in bianco e nero a colori 

numero = 0
for i in range(count):
    numero += 1
    img_path = os.path.join(dirname, 'frames/frame{}.jpg'.format(numero))
    print(img_path)
    net = dnn.readNetFromCaffe(proto_file, model_file)
    kernel = np.load(hull_pts)

    img = cv2.imread(img_path)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    img = cv2.resize(img, (width, height))
    colorized = cv2.resize(colorized, (width, height))

    out_file = os.path.join(r'C:/Users/lucai/Documents/GitHub/video_colorizer-/framec', f'{numero}.jpg')
    cv2.imwrite(out_file, colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()