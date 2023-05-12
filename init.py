import os
import shutil
import pandas as pd
from cv2 import cv2
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.losses import sparse_categorical_crossentropy
from keras import regularizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from matplotlib import pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def get_dataset(directory, filename, input_size):
    images = []

    reader = pd.read_csv(filename)
    encode = LabelEncoder()
    labels = encode.fit_transform(reader['Finding Labels'])
    for index, row in reader.iterrows():
        img = cv2.imread(f'{base_dir}/{row[0]}', cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, input_size)
            # cv2.imwrite(f'D:/Project/PythonProject/Machine Learning/Traditional_food/test/{row[0]}', img)
            images.append(img)

    return np.array(images), np.array(labels)


def train_val_generators(train_image, train_label, validation_image, validation_label, batch_size):

    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow(
        x=train_image,
        y=train_label,
        batch_size=batch_size,
        shuffle=True
    )

    validation_datagen = ImageDataGenerator(rescale=1/255)

    validation_generator = validation_datagen.flow(
        x=validation_image,
        y=validation_label,
        batch_size=batch_size,
    )

    return train_generator, validation_generator


def create_model():

    models = Sequential([
        Conv2D(8, kernel_size=3, padding='same', strides=2, activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(35, activation='softmax')
    ])

    models.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=RMSprop(),
        metrics=['accuracy']
    )
    return models


if __name__ == '__main__':
    BATCH_SIZE = 3
    INPUT_SIZE = (150, 150)
    base_dir = 'D:/Project/PythonProject/Scan-food/Traditional_food/food-tfk-images'

    train_images, train_labels = get_dataset(base_dir, 'train.csv', INPUT_SIZE)
    val_images, val_labels = get_dataset(base_dir, 'dev.csv', INPUT_SIZE)

    print(f"Training images has shape: {train_images.shape} and dtype: {train_images.dtype}")
    print(f"Training labels has shape: {train_labels.shape} and dtype: {train_labels.dtype}")
    print(f"Validation images has shape: {val_images.shape} and dtype: {val_images.dtype}")
    print(f"Validation labels has shape: {val_labels.shape} and dtype: {val_labels.dtype}")

    train_gen, validation_gen = train_val_generators(
        train_images, train_labels,
        val_images, val_labels, BATCH_SIZE
    )

    print(f"Images of training generator have shape: {train_gen.x.shape}")
    print(f"Labels of training generator have shape: {train_gen.y.shape}")
    print(f"Images of validation generator have shape: {validation_gen.x.shape}")
    print(f"Labels of validation generator have shape: {validation_gen.y.shape}")

    model = create_model()

    input('check point')

    history = model.fit(
        train_gen, epochs=15,
        validation_data=validation_gen, batch_size=BATCH_SIZE
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()