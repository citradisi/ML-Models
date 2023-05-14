import os
import random
import pandas as pd
from cv2 import cv2
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils import normalize, to_categorical
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten
from matplotlib import pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def split_data(directory, filename, input_size, types):
    dest_dir = os.path.join(directory, types)
    check = ''

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    reader = pd.read_csv(filename)
    for i in range(len(reader)):
        img = reader.iloc[i, 0]
        label = reader.iloc[i, 2]

        for j, unit in enumerate(reader.iloc[i, 3:]):
            if unit == 1:
                class_label = os.path.join(dest_dir, label)
                if not os.path.exists(class_label):
                    os.makedirs(class_label)

                data_images = os.path.join(class_label, img)
                if not os.path.exists(data_images):
                    src = os.path.join(directory, img)
                    dst = os.path.join(class_label, img)
                    pixel = cv2.imread(src)
                    new_image = cv2.resize(pixel, input_size)
                    cv2.imwrite(dst, new_image)
                    check = 'data successfully sorted'
                else:
                    check = 'the data is already sorted'
    print(check)


def get_data(directory):
    data_ = []

    for index, img_class in enumerate(os.listdir(directory)):
        path = os.path.join(directory, img_class)
        for path_img in os.listdir(path):
            img_path = os.path.join(path, path_img)
            images = cv2.imread(img_path, cv2.IMREAD_COLOR)
            data_.append([images, index])

    return data_


def split_to_feature_and_labels(data_):
    feature = []
    label = []
    for feature_, label_ in data_:
        feature.append(feature_)
        label.append(label_)

    return np.array(feature), np.array(label)


def train_val_generators(x_train_, y_train_, x_val_, y_val_, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow(
        x=x_train_,
        y=y_train_,
        batch_size=batch_size,
    )

    validation_datagen = ImageDataGenerator(rescale=1 / 255)

    validation_generator = validation_datagen.flow(
        x=x_val_,
        y=y_val_,
        batch_size=batch_size
    )

    return train_generator, validation_generator


def create_model():
    activation = 'sigmoid'
    models = Sequential([
        Conv2D(32, kernel_size=3, activation=activation,
               padding='same', input_shape=(100, 100, 3)),
        MaxPooling2D(),

        Conv2D(10, kernel_size=3, activation=activation, padding='same'),
        MaxPooling2D(),

        Flatten(),
        Dense(35, activation='softmax')
    ])

    models.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=RMSprop(),
        metrics=['accuracy']
    )

    return models


def train_the_model(model_, train_gene, val_gene, batch_size):
    history = model_.fit(
        train_gene, epochs=50,
        validation_data=val_gene, batch_size=batch_size
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


if __name__ == '__main__':
    BATCH_SIZE = 3
    INPUT_SIZE = (100, 100)
    base_dir = 'D:/Project/PythonProject/Scan-food/food-tfk-images'

    ## run this only for first time open this project
    # split_data(base_dir, 'train.csv', INPUT_SIZE, 'train')
    # split_data(base_dir, 'dev.csv', INPUT_SIZE, 'validation')
    # split_data(base_dir, 'test.csv', INPUT_SIZE, 'validation')

    data_train = get_data(f'{base_dir}/train')
    data_val = get_data(f'{base_dir}/validation')
    random.shuffle(data_train)

    x_train, y_train = split_to_feature_and_labels(data_train)
    x_validation, y_validation = split_to_feature_and_labels(data_val)

    print(f"Training images has shape: {x_train.shape} and dtype: {x_train.dtype}")
    print(f"Training labels has shape: {y_train.shape} and dtype: {y_train.dtype}")
    print(f"Validation images has shape: {x_validation.shape} and dtype: {x_validation.dtype}")
    print(f"Validation labels has shape: {y_validation.shape} and dtype: {y_validation.dtype}")

    train_gen, validation_gen = train_val_generators(
        x_train, y_train,
        x_validation, y_validation,
        BATCH_SIZE
    )

    print(f"Images of training generator have shape: {train_gen.x.shape}")
    print(f"Labels of training generator have shape: {train_gen.y.shape}")
    print(f"Images of validation generator have shape: {validation_gen.x.shape}")
    print(f"Labels of validation generator have shape: {validation_gen.y.shape}")

    model = create_model()
    # model.summary()
    train_the_model(model, train_gen, validation_gen, BATCH_SIZE)
