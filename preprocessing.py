import os
import shutil
import cv2
import argparse
import subprocess
from skimage.metrics import structural_similarity
from glob import glob
from tqdm import tqdm


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video-path', default='/Users/leon/Downloads/videos/swim.mp4', help='video path')
    parser.add_argument('--save-path', default='/Users/leon/Downloads/swim', help='folder for saving frames')

    args = parser.parse_args()
    return args


def ffmpeg_video(video_path):
    subprocess.call("ffmpeg -i {} -vf 'scale=960:540, fps=5' temp/%08d.png".format(video_path), shell=True)


def remove_repetitive(threshold=0.9):
    # load all frames in temp folder
    images = sorted(glob('temp/*.png'))

    # processd_images
    processed_images = [images[0]]

    # iterate
    img1 = cv2.imread(images[0])
    for i in tqdm(range(1, len(images))):
        img2 = cv2.imread(images[i])

        # compute ssim
        ssim = structural_similarity(img1, img2, multichannel=True)

        if ssim > threshold:
            continue
        else:
            processed_images.append(images[i])
            img1 = img2

    return processed_images


def post_process(processed_images, save_path):
    for file in processed_images:
        shutil.move(file, save_path)
    shutil.rmtree('./temp')


if __name__ == '__main__':
    args = get_config()

    # create temp folder and output folder
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists('./temp'):
        os.mkdir('./temp')

    # extract frames
    print("extracting frames from videos...")
    ffmpeg_video(video_path=args.video_path)

    # remove
    images = remove_repetitive()

    # post processing
    post_process(images, args.save_path)
