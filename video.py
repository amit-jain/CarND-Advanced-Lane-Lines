import glob
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import matplotlib.pyplot as plt

from Calibrate import calibrate
from LaneLineProcessor import process_image


# Extract image from a given video of the given frame
def extract_img_from_video(challenge_input, out_path, frame=1):
    clip2 = VideoFileClip(challenge_input)
    challenge_img = clip2.get_frame(frame)
    plt.imshow(challenge_img)
    plt.show()
    challenge_img = cv2.cvtColor(challenge_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path + 'challenge' + str(frame) + '.jpg', challenge_img)


# Process videos
def process():
    # calibrate camera
    calibrate()

    # process videos
    videos = glob.glob('*_video.mp4')
    for idx, path in enumerate(videos):
        print('Processing video', path)
        clip1 = VideoFileClip(path)
        path = path.split('.')[0]

        video_clip = clip1.fl_image(process_image)
        video_clip.write_videofile(path + '_tracked.mp4', audio=False)

process()
#extract_img_from_video('challenge_video.mp4', "./test_images/", frame=1)
#extract_img_from_video('challenge_video.mp4', "./test_images/", frame=15)