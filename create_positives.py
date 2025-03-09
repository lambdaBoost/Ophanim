import cv2
import numpy as np
import random
import glob
import os

IMG_SIZE = 128 #size of image to be used by cnn
MAX_RANDOM = 200 #max random noise (max is 255)
BLUR_MASK = [3,4] #possible values for the blur mask

SLICED_IMAGE_FOLDER = "train-data/ukraine/pos" #folder of raw images to add mask to
OUTPUT_FOLDER = "train-data/ukraine" #this will be our final training data
POS_FRACTION = 0



#mask images folder
images = glob.glob("synthetic-data/*.png")

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

i=0
for fl in os.listdir(SLICED_IMAGE_FOLDER):
    randchoice = random.uniform(0,1)
    
    if randchoice > POS_FRACTION:
        
        try:
            random_mask = random.choice(images)

            mask = cv2.imread(random_mask)
            mask = mask[:,:,0]

            #add noise
            mask_shape = np.shape(mask)
            mask_x = mask_shape[0]
            mask_y = mask_shape[1]
            random_noise = np.random.randint(MAX_RANDOM, size=mask_shape, dtype = np.uint8)
            mask = cv2.subtract(mask, random_noise)

            #blur
            blurx = random.choice(BLUR_MASK)
            blury = random.choice(BLUR_MASK)
            mask = cv2.blur(mask,(3,3))

            #rotate
            mask = rotate_image(mask,random.randrange(0,360))

            back = cv2.imread(SLICED_IMAGE_FOLDER + '/' + fl)
            back = back[:,:,0]

            #combine images
            #leave buffer around border for now
            x_placement = random.randint(0 + mask_x,IMG_SIZE - mask_x)
            y_placement = random.randint(0 + mask_y,IMG_SIZE - mask_y)

            combined = back.copy()
            combined[x_placement:x_placement+mask_x, y_placement:+y_placement+mask_y] = cv2.add(back[x_placement: x_placement+mask_x, y_placement: y_placement + mask_y] , mask)

            #combined = combined[:,:,0]

            cv2.imwrite(OUTPUT_FOLDER + '/' + 'positive/'+ fl, combined)
        except:
            print('issue with image' + fl)
    
    else:
        try:
            back = cv2.imread(SLICED_IMAGE_FOLDER + '/' + fl)
            back = back[:,:,0]
            cv2.imwrite(OUTPUT_FOLDER + '/' + 'negative/'+ fl, back)

        except:
            print('issue with image' + fl)

    i=i+1
    if i%1000==0:
        print(i)
