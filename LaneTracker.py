import numpy as np
import cv2


class Tracker():
    def __init__(self, window_width, window_height, margin, xm=1, ym=1, smooth_factor=15):
        # list that stores all the past (left, right) center set values used for smoothing the output
        self.recent_centers = []

        # the window pixel width of the center values, used to count pixels inside center windows to determine curve
        #  values
        self.window_width = window_width

        # the window pixel height of the center values, used to count pixels inside center windows to determine curve
        #  values. breaks the image into vertical levels
        self.window_height = window_height

        # the pixel distance in both directions to slide (left_window + right_window) template for searching
        # average x values of the fitted line over the last n iterations
        self.margin = margin

        # meters per pixel in horizontal axis
        self.xm_per_pix = xm

        # meters per pixel in vertical axis
        self.ym_per_pix = ym

        # smooth factor
        self.smooth_factor = smooth_factor

    def find_window_centroids(self, warped):
        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image
        #  slice and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - self.window_width / 2
        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - self.window_width / 2 + int(warped.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(warped.shape[0] / self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(
                warped[int(warped.shape[0] - (level + 1) * self.window_height):int(warped.shape[0] - level *
                                                                                   self.window_height), :], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference Use window_width/2 as offset
            # because convolution signal reference is at right side of window, not center of window
            offset = self.window_width / 2
            l_min_index = int(max(l_center + offset - self.margin, 0))
            l_max_index = int(min(l_center + offset + self.margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - self.margin, 0))
            r_max_index = int(min(r_center + offset + self.margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))

        self.recent_centers.append(window_centroids)
        return np.average(self.recent_centers[-self.smooth_factor:], axis=0)

    def draw_rectangles(self, window_centroids, binary):
        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(binary)
            r_points = np.zeros_like(binary)

            leftx = []
            rightx = []

            # Go through each level and draw the windows
            for level in range(0, len(window_centroids)):
                leftx.append(window_centroids[level][0])
                rightx.append(window_centroids[level][1])

                # Window_mask is a function to draw window areas
                l_mask = window_mask(self.window_width, self.window_height, binary, window_centroids[level][0], level)
                r_mask = window_mask(self.window_width, self.window_height, binary, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | (l_mask == 1)] = 255
                r_points[(r_points == 255) | (r_mask == 1)] = 255

            # Draw the results
            # add both left and right window pixels together
            template = np.array(r_points + l_points, np.uint8)

            # create a zero color channel
            zero_channel = np.zeros_like(template)

            # make window pixels green
            template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)

            # making the original road pixels 3 color channels
            warpage = np.array(cv2.merge((binary, binary, binary)),
                               np.uint8)

            # overlay the orignal road image with window results
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

        # If no window centers found, just display original road image
        else:
            output = np.array(cv2.merge((binary, binary, binary)), np.uint8)

        return output, leftx, rightx

    def curvature(self, image, binary, leftx, rightx, Minv):
        img_size = (image.shape[1], image.shape[0])

        yvals = range(0, binary.shape[0])
        res_yvals = np.arange(binary.shape[0] - (self.window_height / 2), 0, -self.window_height)

        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_fitx = left_fit[0] * yvals * yvals + left_fit[1] * yvals + left_fit[2]
        left_fitx = np.array(left_fitx, np.int32)

        right_fit = np.polyfit(res_yvals, rightx, 2)
        right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
        right_fitx = np.array(right_fitx, np.int32)

        left_lane = np.array(
            list(zip(np.concatenate((left_fitx - self.window_width / 2, left_fitx[::-1] + self.window_width / 2), axis=0),
                     np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

        right_lane = np.array(
            list(zip(np.concatenate((right_fitx - self.window_width / 2, right_fitx[::-1] + self.window_width / 2), axis=0),
                     np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

        middle_marker = np.array(
            list(zip(np.concatenate((left_fitx - self.window_width / 2, right_fitx[::-1] + self.window_width / 2), axis=0),
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
        ym_per_pix = self.ym_per_pix
        xm_per_pix = self.xm_per_pix

        curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32) * ym_per_pix,
                                  np.array(leftx, np.float32) * xm_per_pix, 2)
        curverad = ((1 + (2 * curve_fit_cr[0] * yvals[-1] * ym_per_pix + curve_fit_cr[1]) ** 2) ** 1.5) / \
                   np.absolute(2 * curve_fit_cr[0])

        camera_center = (left_fitx[-1] + right_fitx[-1]) / 2
        center_diff = (camera_center - binary.shape[1] / 2) * xm_per_pix
        side_pos = 'left'
        if center_diff <= 0:
            side_pos = 'right'

        cv2.putText(result, "Curvature Radius = " + str(round(curverad, 3)) + '(m)', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result,
                    "Distance from Center =  " + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return result


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output