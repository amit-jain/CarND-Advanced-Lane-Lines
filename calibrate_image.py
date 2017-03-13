import numpy as np
import cv2
import glob
import pickle


objp = np.zeros((9*6, 3), np.float32)
print(objp.shape)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
print(objp.shape)

objpoints = []
imgpoints = []

calibration_imgs = glob.glob('camera_cal/calibration*.jpg')

for idx, path in enumerate(calibration_imgs):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret ==  True:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        write_name = 'corners_found' + str(idx) + '.jpg'
        cv2.imwrite('./output_images/' + write_name, img)

img = cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
dist_pickle = {}
dist_pickle ['matrix'] = cameraMatrix
dist_pickle['distances'] = distCoeffs
pickle.dump(dist_pickle, open('./calibration_pickle.p', 'wb'))