import keras
from keras import layers
from keras.models import Sequential
#from matplotlib import pyplot as plt
import numpy as np
import os

BATCH_SIZE = 256
IMAGE_SIZE = 128
EPOCHS = 25
LR = 0.001

#disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_dataset, val_dataset = keras.utils.image_dataset_from_directory("train-data/ukraine",
                                                         validation_split=0.2,
                                                         subset = 'both',
                                                         seed = 57,
                                                         image_size = (128,128),
                                                         color_mode = "grayscale",
                                                         batch_size = BATCH_SIZE)


#sense-check
#sense-check
"""
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(10):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
"""

def build_cnn(image_size, num_classes = 2):

    #rescale
    inputs = keras.Input(shape= (image_size,image_size,1))

    x = inputs
    x = layers.Rescaling(1./255)(x)

    x = layers.Conv2D(2, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(8, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(256, (3,3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Flatten()(x)

#TODO- the main improvement came from sigmoid layers - hardly a surprise
    x = layers.Dense(256, activation = 'sigmoid')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation = 'sigmoid')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(32, activation = 'sigmoid')(x)

    x = layers.Dense(8, activation = 'sigmoid')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    

    return keras.Model(inputs, outputs)

def build_simple_cnn(img_size, num_classes):
    """
    a much simpler model which seems to actually perform
    """
    model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_size, img_size, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation = 'sigmoid')])

    return model



cnn = build_cnn(IMAGE_SIZE, 2)

cnn.compile(
    optimizer=keras.optimizers.Adam(LR),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")]
)
cnn.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset)


cnn.save("full_model_ukraine2.keras")
results = cnn.evaluate(val_dataset)

