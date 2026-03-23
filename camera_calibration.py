import cv2
import numpy as np
import glob
import os
import pickle

def calibrate_camera():
    # checkerboard dimensions (internal corners)
    CHECKERBOARD = (8, 6)
    
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 
                            0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image
    
    images = glob.glob('calibration_images/*.jpg')
    print(f"Found {len(images)} calibration images")
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            # refine corner locations to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + 
                       cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), 
                                       criteria)
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"  Found corners in {fname}")
    
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print(f"\nReprojection error: {ret:.4f} px (should be < 1.0)")
    print(f"Calibration image size: {gray.shape[::-1]}")
    print(f"\nCamera matrix K:\n{K}")
    print(f"\nDistortion coefficients:\n{dist}")
    
    # save for later use, including the calibration image dimensions
    with open('calibration.pkl', 'wb') as f:
        pickle.dump({'K': K, 'dist': dist, 'cal_size': gray.shape[::-1]}, f)
    
    return K, dist

K, dist = calibrate_camera()