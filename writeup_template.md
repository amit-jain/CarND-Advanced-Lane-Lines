**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration.png "Calibration"
[image2]: ./output_images/undistort.png "Undistorted"
[image3]: ./output_images/warped.png "Warped"
[image4]: ./output_images/binary.png "Binary Challenge"
[image5]: ./output_images/binary_test.png "Binary"
[image6]: ./output_images/tracked_test.png "Tracked"
[image7]: ./output_images/tracked.png "Tracked Challenge"
[video1]: ./project_video_tracked.mp4 "Project Video"
[video2]: ./challenge_video_tracked.mp4 "Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

My project includes the following files:
* LaneLineProcessor.py main class that builds the pipeline through different steps
* calibration.py class that calibrates the camera images using the calibrating images
* LaneTracker.py class has methods for finding lane line centroids, curvature and visualization of images
* video.py code to open and save videos passing each frame thought the processing pipeline
* writeup_report.md report summarizing the results

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `Calibrate.py` in the method `calibrate`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
 Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for
  each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended 
  with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be 
  appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard 
  detection. The following image shows the corners detected for each of the image.

![image1]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients 
using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test images using the `cv2
.undistort()` function and obtained this result: 

![image2]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
As described above following is the example of the distortion correction applied to the test image. The 
following screen shot shows the original image and corrected image after distortion correction:

![image2]

####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I perform perspective transformation on the main image before binary thresholding as it seemed to perform well in 
experiments. 
The code for my perspective transform includes a function called `birds_eye_perspective()`, which appears in lines 
94 through 104 in the file `LaneLineProcessor.py` (./LaneLineProcessor.py).  The `perspective()` function takes as 
input an image (`img`).  I hardcoded the source and destination points in the following manner:

```
    src = np.float32([[490, 482], [810, 482],
                      [1240, 720], [40, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                     [1160, 720], [120, 720]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 490, 482      | 0, 0          | 
| 810, 482      | 1280, 0       |
| 1240, 720     | 1160, 720     |
| 40, 720       | 120, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test 
image and its warped counterpart to verify that the lines appear parallel in the warped image.

![image3]

####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The combination of thresholding is 
done in the method `apply_mask` (Lines 80 - 91) in `LabeLineProcessor.py` (./LaneLineProcessor.py). The various other
 gradient and color threshold methods are `abs_sobel_thresh`, `mag_thresh`, `dir_threshold` and  `color_threshold`.
  Here's an example of the output for this step.  (note: the first one is an image extracted from the challenge 
  video)
  
![image4]

![image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for identifying lane lines is in `find_window_centroids` method in the class `LaneTracker.py` (./LaneTracker
.py).
I used the sliding windows approach using convolutions as descibed in the nodes to find best window center positions 
to identify lanes. This approach works by sliding a small window kernel/template across the image from left to right and any 
overlapping values are summed together, creating the convolved signal. The peak of the convolved signal is where 
there was the highest overlap of pixels and the most likely position for the lane marker.

The method `draw_rectangles` (Lines 70 - 114) in `LaneTracker.py` is then used to draw these rectangles over the 
image for visualization.

The a second order polynomial was fit to the these identified points and a curve drawn which would depict the tracked
 lanes. 
 
####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The polynomial fit, the radius of curvature of the lane and the positions of the vehicle with respect to the center 
is done in the method `curvature` in the file `LaneTracker.py` in line numbers 116 - 175.


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The result of the lane identification and tracking operation is available in the following images. The first image is
 one of the test images provided and the lane tracking works quite well here. 

![image6]

On the other hand while trying to correctly track lanes in the challenge video was only a partial success. The 
following image shows the lane tracking result on an extracted image from the challenge video.
![image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
The `video.py` calls the `process_image` function of the `LaneLineProcessor` for each frame of the video and 
constructs the video from the processed output images.

Here's a link to the result on the [Project video](./project_video_tracked.mp4)
and [Challenge video](./challenge_video_tracked.mp4) which is a decent attempt but can be improved a lot.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline worked well on the project video but still has problems with the challenge video. The problems faced 
particularly are:
* Many frames have missing lane markers for most of the eithe of the lanes (quite frequently observed in the challenge 
video)
* Many frames (even on the project video) have quite a few straight gradient changes which makes it hard to 
correctly target the actual lane lines with trial and error on thresholds.

The pipeline can be further improved in the tracker when doing sliding window convolutions where the starting point 
is the previous detection found. Even though I added smoothing to the centroids but found it didn't quite help. There 
are other normalization techniques that can be explored like histogram equalization etc. to further reduce the wide 
variations in lightening and gradients to help come to better thresholds for binary image detection. The other 
point that can be considered is explicit outlier detection and removal. Though some of the effect should have been 
minimized with smoothing but it may not have been enough.

The pipeline performs very poorly in the harder challenge video and fails to take into account the sharp curves. Not 
really sure what techniques can be applied but a place to start would be moving away from hardcoded source and 
destinations points for warping.