
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import math
import os
from Utils.MiscUtils import *
from Utils.ImageUtils import *
from Utils.MathUtils import *
import scipy.optimize 
import argparse
square_side = 12.5


def run(mode):

    if (mode == "ir"):
        folder_name = r'C:\Users\Asus\Documents\Shuna Ni\images\ir_mod'   #folder where IR images are stored
        save_folder = r"C:\Users\Asus\Documents\Shuna Ni\images\ir_saved"  #folder where IR images are to be saved
    elif (mode=="rgb"):
        folder_name = r'C:\Users\Asus\Documents\Shuna Ni\images\rgb_mod'    #folder where RGB images are stored
        save_folder = r"C:\Users\Asus\Documents\Shuna Ni\images\rgb_saved"  #folder where IR images are to be saved

    images = loadImages(folder_name)
    h, w = [6,7]
    all_image_corners = getImagesPoints(images, h, w)
    print(len(all_image_corners) , len(all_image_corners[0]))
    world_corners = getWorldPoints(square_side, h, w)

    displayCorners(images, all_image_corners, h, w, save_folder)

    print("Calculating H for %d images", len(images))
    all_H_init = getAllH(all_image_corners, square_side, h, w)
    print("Calculating B")
    B_init = getB(all_H_init)
    print("Estimated B = ", B_init)
    print("Calculating A")
    A_init = getA(B_init)
    print("Initialized A = ",A_init)
    print("Calculating rotation and translation")
    all_RT_init = getRotationAndTrans(A_init, all_H_init)
    print("Init Kc")
    kc_init = np.array([0,0]).reshape(2,1)
    print("Initialized kc = ", kc_init)


    return A_init, all_image_corners, all_H_init, world_corners


#run once

rgb_A, rgb_corners, rgb_h, wc = run("rgb")
ir_A, ir_corners, ir_h, wc_1 = run("ir")

# reprojects control points seen by the RGB camera onto the IR camera's frame of reference and finds distance 


rgb_to_ir = []
#calculates the homogenous transformation matrix
for i in range(len(rgb_h)):
    H_rgb_inv = np.linalg.inv(rgb_h[i])
    H_ir = ir_h[i]
    rgb_to_ir.append(np.matmul(H_ir, H_rgb_inv))

# print(H_inv.shape, rgb_corners[0][0])
error_by_image = []
all_projections = []
for i in range(len(rgb_corners)):
    projections = []
    error = 0
    for j in range(len(rgb_corners[i])):
        rgb_corner = rgb_corners[i][j]
        corner2 = np.array([rgb_corner[0],rgb_corner[1],1]).reshape(3,1)
        calculated_point_3 = np.matmul(rgb_to_ir[i],corner2)
        calculated_point_2 = np.array([calculated_point_3[0][0]/calculated_point_3[2][0],calculated_point_3[1][0]/calculated_point_3[2][0] ])
        projections.append(calculated_point_2)
        actual_detected_corner = ir_corners[i][j]
        
        error += np.linalg.norm(calculated_point_2 - actual_detected_corner)
    all_projections.append(projections)
    mean_error = error/j
    error_by_image.append(mean_error) 

print("Re-projection error for each image", error_by_image)
print("Mean error", sum(error_by_image)/len(error_by_image))

open_folder = r"C:\Users\Asus\Documents\Shuna Ni\images\ir_mod" #Enter folder where IR images are stored
save_folder = r"C:\Users\Asus\Documents\Shuna Ni\images\reproj_ir" #Enter folder where reprojected point images are to be stored
images = loadImages(open_folder)
for i,image in enumerate(images):
    drawn = cv2.drawChessboardCorners(image, (7, 6), ir_corners[i], True)
    # cv2.imshow('Chessboard',drawn)
    # cv2.waitKey(0)
    for point in all_projections[i]:
        x = int(point[0])
        y = int(point[1])
        image = cv2.circle(drawn, (x, y), 5, (0, 0, 255), 1)
        
    filename = save_folder + "\\" + str(i) + "reproj.png"
    cv2.imwrite(filename, image)
