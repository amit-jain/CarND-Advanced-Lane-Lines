import glob
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
import matplotlib.pyplot as plt

from Calibrate import calibrate
from LaneLineProcessor import Processor


# Extract image from a given video of the given frame
def extract_img_from_video(input, out_path, time=1):
    vidcap = cv2.VideoCapture(input)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)  # just cue to 20 sec. position
    success, image = vidcap.read()

    plt.imshow(image)
    plt.show()
    # challenge_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_path + 'challenge' + str(time) + '.jpg', image)


# Process videos
def process():
    # calibrate camera
    calibrate()

    # Initalize the processor and tracker
    processor = Processor()

    # process videos
    videos = glob.glob('*_video.mp4')
    for idx, path in enumerate(videos):
        print('Processing video', path)
        clip1 = VideoFileClip(path)
        path = path.split('.')[0]

        video_clip = clip1.fl_image(processor.process_image)
        video_clip.write_videofile(path + '_tracked.mp4', audio=False)


process()

# extract_img_from_video('project_video.mp4', "./test_images/", time=30)
# extract_img_from_video('project_video.mp4', "./test_images/", time=23)
# extract_img_from_video('project_video.mp4', "./test_images/", time=19)
# extract_img_from_video('project_video.mp4', "./test_images/", time=39)
# extract_img_from_video('project_video.mp4', "./test_images/", time=40)
# extract_img_from_video('project_video.mp4', "./test_images/", time=41)
