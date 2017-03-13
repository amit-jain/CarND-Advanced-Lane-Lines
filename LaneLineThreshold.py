import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from LaneTracker import slide_windows, skip_sliding_windows, visualize, curvature


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    direction = np.arctan2(sobely, sobelx)
    dir_binary = np.zeros_like(direction)
    dir_binary[(direction > thresh[0]) & (direction < thresh[1])] = 1
    return dir_binary


def color_threshold(img, sthresh=(0, 255), vthresh=(0, 255)):
    # Convert to HLS color space
    # Apply a threshold to the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    binary_s = np.zeros_like(s)
    binary_s[(s > sthresh[0]) & (s <= sthresh[1])] = 1

    # Convert to HSV color space
    # Apply a threshold to the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    binary_v = np.zeros_like(v)
    binary_v[(v > vthresh[0]) & (v <= vthresh[1])] = 1

    # Return a binary image of threshold result from both s channel of HLS and v channel of HSV
    output = np.zeros_like(s)
    output[(binary_s == 1) & (binary_v == 1)] = 1
    return output

def project(undist, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)

dist_pickle = pickle.load(open('calibration_pickle.p', 'rb'))
matrix = dist_pickle['matrix']
distances = dist_pickle['distances']

images = glob.glob('./test_images/*.jpg')

for idx, path in enumerate(images):
    image = cv2.imread(path)

    # undistort the image
    image = cv2.undistort(image, matrix, distances, None, matrix)

    # Write the undistorted output
    cv2.imwrite('./output_images/undistort' + str(idx) + '.jpg', image)

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))
    color_binary = color_threshold(image, sthresh=(100, 255), vthresh=(100, 255))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (color_binary == 1)] = 255

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #plt.show()

    # Work on defining prospective transform
    img_size = (image.shape[1], image.shape[0])
    bottom_width = 0.7
    top_width = 0.08
    height_pct = 0.62
    bottom_trim = 0.935
    src = np.float32([[image.shape[1]*(0.5 - top_width/2), image.shape[0] * height_pct],
                     [image.shape[1]*(0.5 + top_width/2), image.shape[0] * height_pct],
                     [image.shape[1]*(0.5 - bottom_width/2), image.shape[0] * bottom_trim],
                     [image.shape[1]*(0.5 + bottom_width/2), image.shape[0] * bottom_trim]])
    offset = image.shape[1] * 0.25

    dst = np.float32([[offset, 0],
                      [image.shape[1] - offset, 0],
                      [offset, image.shape[0]],
                      [image.shape[1] - offset, image.shape[0]]
                      ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)
    cv2.imwrite('./output_images/warped' + str(idx) + '.jpg', warped)

    left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = slide_windows(warped)
    result = visualize(warped, left_fit, right_fit, nonzerox, nonzeroy ,left_lane_inds, right_lane_inds)
    cv2.imwrite('./output_images/lanemarked' + str(idx) + '.jpg', result)

    print(curvature(img_size, left_fit, right_fit))
