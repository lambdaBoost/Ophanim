# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:43:26 2019

this script will slice a directory of large tiff images into inputs for the CNN
can be used for both training and inference data

@author: ahall
"""

import numpy as np
import cv2
import os
import random
from PIL import Image
# from shutil import copyfile

Image.MAX_IMAGE_PIXELS = 933120000

X_SIZE = 128 # 120
Y_SIZE = 128 # 90
SOURCE_DIR = "./inference_data/north-caucus"  # noqa E501
DEST_DIR = "./inference_data/sliced/north-caucus-set5"  # noqa E501
TEST_PROP = 0.0

# returns the binary image


def convert_to_binary(input_file, size_x, size_y):
    input_img = cv2.imread(input_file)
    input_img = cv2.resize(input_img, (size_x, size_y))

    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    #mask = cv2.inRange(hsv, (85, 30, 20), (130, 255, 255))
    mask = cv2.inRange(hsv, (70, 50, 50), (140, 255, 255))
    inv_mask = cv2.bitwise_not(mask)

    #inv_mask = mask <= 0
    #blue = np.zeros_like(input_img, np.uint8)
    #blue[inv_mask] = input_img[inv_mask]

    #thresh = 64
    #grey = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    #binary = cv2.threshold(grey, thresh, 255, cv2.THRESH_BINARY)[1]

    return inv_mask


def slice_images(
        source_directory,
        dest_directory,
        fileName,
        height,
        width,
        test_prop):

    im = Image.open(os.path.join(source_directory, fileName))
    imgwidth = im.size[0]
    imgheight = im.size[1]
    for i in range(0, imgheight, int(height/2)):
        print(fileName)
        print(i)
        
        for j in range(0, imgwidth, int(width/2)):
            box = (j, i, j + width, i + height)

            try:
                a = im.crop(box)
                if(i + height < imgheight and j + width < imgwidth):
                    if test_prop == 0:
                        a.save(
                            os.path.join(
                                dest_directory,
                                f"{fileName}-{i}-{j}.png" ))

                    elif(random.uniform(0, 1) < test_prop):
                        a.save(
                            os.path.join(
                                dest_directory,
                                "test",
                                f"{fileName}-{i}-{j}.png" ))
                    else:
                        a.save(
                            os.path.join(
                                dest_directory,
                                "train",
                                f"{fileName}-{i}-{j}.png" ))
            except BaseException:
                pass


# processes all the raw images in a given directory
def process_all_images(
        source_directory,
        dest,
        size_x,
        size_y,
        test_prop,
        slice_img):
    i = 0
    # for root, dirs, files in os.walk(source_directory):
    for image in os.listdir(source_directory):

        print(i)
        rand = random.uniform(0, 1)

        if(slice_img):

            slice_images(
                source_directory,
                dest,
                image,
                size_y,
                size_x,
                test_prop)


        i = i + 1


if __name__ == "__main__":
    process_all_images(SOURCE_DIR, DEST_DIR, X_SIZE, Y_SIZE, TEST_PROP, True)
