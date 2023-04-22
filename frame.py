from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np
import cv2
import os
from cv2 import dnn
from PIL import Image

relative_path = os.path.dirname(__file__)

def colorizeFrames(height, width, black_and_white_frames_path, colored_frames_path, frame_count):
    proto_file = os.path.join(relative_path, 'colorization_deploy_v2.prototxt')
    model_file = os.path.join(relative_path, 'colorization_release_v2.caffemodel')
    hull_pts = os.path.join(relative_path, 'pts_in_hull.npy')
    
    index = 0
    for i in range(frame_count):
            index += 1
            img_path = os.path.join(black_and_white_frames_path, 'frame{}.jpg'.format(index))
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

            out_file = os.path.join(colored_frames_path, f'{index}.jpg')
            print(out_file)
            cv2.imwrite(out_file, colorized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def separateAudioTrack(video_path):
    audio_path = os.path.join(relative_path, 'tmp/audio.wav')
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)
    audio_clip.close()
    video_clip.close()


def main():
    video_path = os.path.join(relative_path, 'video.mp4')
    black_and_white_frames_path = os.path.join(relative_path, 'tmp/black_and_white_frames')
    colored_frames_path = os.path.join(relative_path, 'tmp/colored_frame')

    if (not os.path.exists(video_path)):
        print("video.mp4 not found!")
        return 0

    if (not os.path.exists("tmp")):
        os.mkdir("tmp")

    if (not os.path.exists(black_and_white_frames_path)):
        os.mkdir("tmp/black_and_white_frames")

    if (not os.path.exists(colored_frames_path)):
        os.mkdir("tmp/colored_frame")

    print("Separating audio from mp4")
    separateAudioTrack(video_path)

    # converte i frames in bianco e nero in frame
    print("\nConverting video to frames")
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = 0
    while success:
        cv2.imwrite(os.path.join(black_and_white_frames_path, f"frame{frame_count}.jpg"), image)
        success, image = vidcap.read()
        frame_count += 1

        if cv2.waitKey(10) == 27:
            break
    
    print("Done.")
    
    vidcap.release()
    frame = cv2.imread(os.path.join(black_and_white_frames_path, "frame0.jpg"))
    height, width, _ = frame.shape

    frame_count -= 1
    print("\nFrame number " + str(frame_count))

    print("\nColorizing frames ")
    colorizeFrames(height, width, black_and_white_frames_path, colored_frames_path, frame_count)
    print("Done.")

if __name__ == '__main__':
    main()
