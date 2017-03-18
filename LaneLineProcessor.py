import cv2
import numpy as np
import pickle
import glob

from Calibrate import calibrate
from LaneTracker import Tracker


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direction = np.arctan2(sobely, sobelx)
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return dir_binary


def color_threshold(img, thresh_b=(155, 225), thresh_l=(210, 255)):
    # Generate binary thresholded images
    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 2]
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 0]

    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= thresh_b[0]) & (b_channel <= thresh_b[1])] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= thresh_l[0]) & (l_channel <= thresh_l[1])] = 1

    combined_binary = np.zeros_like(b_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    return combined_binary


def apply_mask(image, ksize=3):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(40, 160))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(40, 160))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 140))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.8, 1.3))
    color_binary = color_threshold(image)

    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | (color_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))] = 255

    return combined


def birds_eye_perspective(image):
    img_size = (image.shape[1], image.shape[0])

    src = np.float32([[490, 482], [810, 482],
                      [1240, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                      [1160, 720], [120, 720]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

    return M, Minv, warped


class Processor():
    def __init__(self):
        # list that stores all the past (left, right) center set values used for smoothing the output
        self.tracker = Tracker(window_width=20, window_height=120, margin=25, xm=3.7/700, ym=30/720,
                                smooth_factor=20)

        dist_pickle = pickle.load(open('calibration_pickle.p', 'rb'))
        self.matrix = dist_pickle['matrix']
        self.distances = dist_pickle['distances']

    matrix = []
    distances = []
    output_path = './output_images/'

    # Input expects a RGB image
    def process_image(self, image, path=None):
        # Convert to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Undistort the image
        image = cv2.undistort(image, self.matrix, self.distances, None, self.matrix)
        if path:
            cv2.imwrite(self.output_path + path + '_distort.jpg', image)

        # Warp the image
        M, Minv, warped = birds_eye_perspective(image)
        if path:
            cv2.imwrite(self.output_path + path + '_warped.jpg', warped)

        # Apply thresholding functions
        binary = apply_mask(warped)
        if path:
            cv2.imwrite(self.output_path + path + '_binary.jpg', binary)

        # Get the lane window centroids and draw them
        window_centroids = tracker.find_window_centroids(binary)

        lanemarked, leftx, rightx = tracker.draw_rectangles(window_centroids, binary)
        if path:
            cv2.imwrite(self.output_path + path + '_lanemarked.jpg', lanemarked)

        # Track the lanes and compute curvature and position
        tracked = tracker.curvature(image, binary, leftx, rightx, Minv)
        if path:
            cv2.imwrite(self.output_path + path + '_tracked.jpg', tracked)

        # Convert the result back to RGB
        result = cv2.cvtColor(tracked, cv2.COLOR_BGR2RGB)

        return result


# Calibrate the camera
calibrate()

# Initialize the processor
processor = Processor()

# process test images
images = glob.glob('./test_images/*.jpg')
for idx, path in enumerate(images):
    # Initialize tracker for each image
    tracker = Tracker(window_width=20, window_height=120, margin=35, xm=3.7/700, ym=30/720, smooth_factor=20)
    processor.tracker = tracker
    image = cv2.imread(path)
    path = path.split('/')[2].split('.')[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processor.process_image(image, path)
