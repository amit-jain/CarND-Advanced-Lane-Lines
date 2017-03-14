import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob

from moviepy.video.io.VideoFileClip import VideoFileClip

from tracker import Tracker, window_mask

dist_pickle = pickle.load(open('calibration_pickle.p', 'rb'))
matrix = dist_pickle['matrix']
distances = dist_pickle['distances']


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


def color_threshold(img, sthresh=(0, 255), vthresh=(0, 255)):
    # Convert to HLS color space
    # Apply a threshold to the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s = hls[:, :, 2]
    binary_s = np.zeros_like(s)
    binary_s[(s > sthresh[0]) & (s <= sthresh[1])] = 1

    # Convert to HSV color space
    # Apply a threshold to the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    binary_v = np.zeros_like(v)
    binary_v[(v > vthresh[0]) & (v <= vthresh[1])] = 1

    # Return a binary image of threshold result from both s channel of HLS and v channel of HSV
    output = np.zeros_like(s)
    output[(binary_s == 1) & (binary_v == 1)] = 1
    return output


# def project(undist, ploty, left_fitx, right_fitx):
#     # Create an image to draw the lines on
#     warp_zero = np.zeros_like(warped).astype(np.uint8)
#     color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
#     # Recast the x and y points into usable format for cv2.fillPoly()
#     pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#     pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#     pts = np.hstack((pts_left, pts_right))
#
#     # Draw the lane onto the warped blank image
#     cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
#
#     # Warp the blank back to original image space using inverse perspective matrix (Minv)
#     newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
#     # Combine the result with the original image
#     result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
#     plt.imshow(result)


def process_image(image, idx=-1):
    # undistort the image
    image = cv2.undistort(image, matrix, distances, None, matrix)

    if idx != -1:
        # Write the undistorted output
        cv2.imwrite('./output_images/undistort' + str(idx) + '.jpg', image)

    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

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
    # plt.show()

    # Work on defining prospective transform
    img_size = (image.shape[1], image.shape[0])
    bottom_width = 0.7
    top_width = 0.08
    height_pct = 0.62
    bottom_trim = 0.935
    src = np.float32([[image.shape[1] * (0.5 - top_width / 2), image.shape[0] * height_pct],
                      [image.shape[1] * (0.5 + top_width / 2), image.shape[0] * height_pct],
                      [image.shape[1] * (0.5 - bottom_width / 2), image.shape[0] * bottom_trim],
                      [image.shape[1] * (0.5 + bottom_width / 2), image.shape[0] * bottom_trim]])
    offset = image.shape[1] * 0.25

    dst = np.float32([[offset, 0],
                      [image.shape[1] - offset, 0],
                      [offset, image.shape[0]],
                      [image.shape[1] - offset, image.shape[0]]
                      ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)

    if idx != -1:
        cv2.imwrite('./output_images/warped' + str(idx) + '.jpg', warped)

    window_width = 25
    window_height = 80
    tracker = Tracker(window_width=window_width, window_height=window_height, margin=25, xm=4 / 384, ym=10 / 720,
                      smooth_factor=15)

    window_centroids = tracker.find_window_centroids(warped)
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        leftx = []
        rightx = []

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])

            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channle
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((warped, warped, warped)),
                           np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    if idx != -1:
        # Display the final results
        cv2.imwrite('./output_images/lanemarked' + str(idx) + '.jpg', output)

    yvals = range(0, warped.shape[0])
    res_yvals = np.arange(warped.shape[0] - (window_height / 2), 0, -window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(
        list(zip(np.concatenate((left_fitx - window_width / 2, left_fitx[::-1] + window_width / 2), axis=0),
                 np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    right_lane = np.array(
        list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                 np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    middle_marker = np.array(
        list(zip(np.concatenate((left_fitx - window_width / 2, right_fitx[::-1] + window_width / 2), axis=0),
                 np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(image)
    road_bkg = np.zeros_like(image)

    cv2.fillPoly(road, [left_lane], color=[255, 0, 0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [middle_marker], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(image, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)
    ym_per_pix = tracker.ym_per_pix
    xm_per_pix = tracker.xm_per_pix

    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix, np.array(leftx, np.float32) * xm_per_pix, 2)
    curverad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / \
               np.absolute(2 * curve_fit_cr[0])

    camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
    center_diff = (camera_center - warped.shape[1] / 2) * xm_per_pix
    side_pos = 'left'
    if center_diff <= 0:
        side_pos = 'right'

    cv2.putText(result, "Curvature Radius = " + str(round(curverad, 3) + '(m)'), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Distance from Center =  " + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center',
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if idx != -1:
        cv2.imwrite('./output_images/tracked' + str(idx) + '.jpg', output)

    return result


images = glob.glob('./test_images/*.jpg')

for idx, path in enumerate(images):
    image = cv2.imread(path)
    process_image(image)

out_video = 'output1_tracked.mp4'
in_video = 'project_video.mp4'

clip1 = VideoFileClip(in_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(out_video, audio=False)
