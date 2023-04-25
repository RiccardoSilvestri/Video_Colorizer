from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import cv2
import os
from cv2 import dnn
from PIL import Image
import time
from multiprocessing import Pool
from convertito import coloredFramesToVideo

relative_path = os.path.dirname(__file__)

def colorizeSingleFrame(current_frame):
    colored_frames_output_folder = os.path.join(relative_path, 'tmp/colored_frames') #to remove from here --> probably public ?
    black_and_white_frames_folder = os.path.join(relative_path, 'tmp/black_and_white_frames') #same

    proto_file = os.path.join(relative_path, 'res/colorization_deploy_v2.prototxt')
    model_file = os.path.join(relative_path, 'res/colorization_release_v2.caffemodel')
    hull_pts = os.path.join(relative_path, 'res/pts_in_hull.npy')
    
    img_path= os.path.join(black_and_white_frames_folder, f"frame{current_frame}.jpg")
    
    frame = cv2.imread(img_path)
    height, width, _ = frame.shape

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

    out_file = os.path.join(colored_frames_output_folder, f"frame{current_frame}.jpg")
    print(out_file)
    cv2.imwrite(out_file, colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return out_file

def colorizeFrames(frame_number):
    print(frame_number)
    with Pool() as pool:
        results = pool.imap(colorizeSingleFrame, range(frame_number))
        for out_file in results:
            print(out_file)


def separateAudioTrack(video_path):
    audio_path = os.path.join(relative_path, 'tmp/audio.wav')
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)
    audio_clip.close()
    video_clip.close()

def videoToBlackAndWhiteFrames(video_path, black_and_white_frames_folder):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_number = 0
    while success:
        cv2.imwrite(os.path.join(black_and_white_frames_folder, f"frame{frame_number}.jpg"), image)
        success, image = vidcap.read()
        frame_number += 1

        if cv2.waitKey(10) == 27:
            break
        
    vidcap.release()
    return frame_number

def checkFoldersAndModels(black_and_white_frames_folder, colored_frames_output_folder, video_path):
    if (not os.path.exists(video_path)):
        print("video.mp4 not found!")
        return False

    if (not os.path.exists("tmp")):
        os.mkdir("tmp")

    if (not os.path.exists('output')):
        os.mkdir("output")

    if (not os.path.exists(black_and_white_frames_folder)):
        os.mkdir("tmp/black_and_white_frames")

    if (not os.path.exists(colored_frames_output_folder)):
        os.mkdir("tmp/colored_frames")

    if (not os.path.exists('res/colorization_deploy_v2.prototxt')):
        print("res/colorization_deploy_v2.prototxt not found")
        return False
    
    if (not os.path.exists('res/colorization_release_v2.caffemodel')):
        print("res/colorization_release_v2.caffemodel not found")
        return False
    
    if (not os.path.exists('res/pts_in_hull.npy')):
        print("res/pts_in_hull.npy not found")
        return False

def main():
    video_path = os.path.join(relative_path, 'video.mp4')
    black_and_white_frames_folder = os.path.join(relative_path, 'tmp/black_and_white_frames')
    colored_frames_output_folder = os.path.join(relative_path, 'tmp/colored_frames')

    if (checkFoldersAndModels(black_and_white_frames_folder, colored_frames_output_folder, video_path) == False):
        return 0
    
    print("Separating audio from mp4")
    separateAudioTrack(video_path)

    print("\nConverting video to frames")
    frame_number = videoToBlackAndWhiteFrames(video_path, black_and_white_frames_folder)
    print("Done.")

    print("\nFrame number " + str(frame_number))

    print("\nColorizing frames ")
    t0 = time.time()
    colorizeFrames(frame_number)
    t1 = time.time()
    print("Done.")
    print(t1-t0)
    coloredFramesToVideo()

if __name__ == '__main__':
    main()