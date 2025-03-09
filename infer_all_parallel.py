from tensorflow import keras
import random
import os
from matplotlib import pyplot as plt
import numpy as np
import shutil
import asyncio
import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

INFER_DIRECTORY = "./inference_data/sliced/north-caucus"
IMAGE_SIZE = 128
MODEL = "full_model_ukraine2.keras"
OUTPUT_PREDICTIONS = 'NCset6-predictions.pickle'
OUTPUT_FILENAMES = 'NCset6-filenames.pickle'

model = keras.models.load_model(MODEL)
keras.utils.plot_model(model, show_shapes=True)


img_dataset = keras.utils.image_dataset_from_directory(INFER_DIRECTORY,
    shuffle = False,
    labels = None,
    image_size = (128,128),
    color_mode = "grayscale")


predictions = model.predict(img_dataset, verbose = 0)
filenames = img_dataset.file_paths

with open(OUTPUT_PREDICTIONS, 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(OUTPUT_FILENAMES, 'wb') as handle:
    pickle.dump(filenames, handle, protocol=pickle.HIGHEST_PROTOCOL)
