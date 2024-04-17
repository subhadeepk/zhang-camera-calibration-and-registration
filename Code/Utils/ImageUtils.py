import cv2
import numpy as np
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def getImagesPoints(imgs, h, w):
    images = imgs.copy()
    all_corners = []
    print(len(images))
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Find the chess board corners
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        retval, corners = cv2.findChessboardCornersSB(gray, (w, h), flags=flags)
        drawn = cv2.drawChessboardCorners(image, (7, 6), corners, retval)
        # cv2.imshow('Chessboard',drawn)
        # cv2.waitKey(0)
        if retval == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            corners2 = corners2.reshape(-1,2)
            c = corners2[::-1]
            # corners2 = np.hstack((corners2.reshape(-1,2), np.ones((corners2.shape[0], 1))))
            all_corners.append(c)
        # print(all_corners)
    return all_corners

def displayCorners(images, all_corners, h, w, save_folder):
    for i, image in enumerate(images):
        corners = all_corners[i]
        corners = np.float32(corners.reshape(-1, 1, 2))
        cv2.drawChessboardCorners(image, (w, h), corners, True)
        img = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
        # cv2.imshow('frame', img)
        filename = save_folder + "\\" + str(i) + "draw.png"
        cv2.imwrite(filename, img)
        # cv2.waitKey()

    # cv2.destroyAllWindows()