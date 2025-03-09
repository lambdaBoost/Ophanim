from tensorflow import keras
import random
import os
from matplotlib import pyplot as plt
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

INFER_DIRECTORY = "inference_data/sliced/ukraine"
IMAGE_SIZE = 128
MODEL = "full_model-NC.keras"

all_images = os.listdir(INFER_DIRECTORY)

random_image = random.choice(all_images)

img = keras.utils.load_img(os.path.join(INFER_DIRECTORY, random_image), color_mode = 'grayscale')
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

model = keras.models.load_model(MODEL)
predictions = model.predict(img_array)
#score = float(keras.ops.sigmoid(predictions[0][0]))
score = predictions[0][0]
print(f"This image is {100 * (1 - score):.2f}% neg and {100 * score:.2f}% pos.")