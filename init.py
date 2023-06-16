from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
from  keras.applications import VGG19
from keras.models import load_model, Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras import regularizers
from keras.losses import sparse_categorical_crossentropy
from PIL import Image
import os


def make_to_jpeg(src_dir):
    output = 'food-tfk-images/data'
    for class_dir in os.listdir(src_dir):
        path = os.path.join(src_dir, class_dir)
        check = os.path.join(output, class_dir)

        if not os.path.exists(check):
            os.mkdir(check)

        for i, img in enumerate(os.listdir(path)):
            image = Image.open(os.path.join(path, img))
            dest_path = os.path.join(check, f'{class_dir}{i}.jpg')
            rgb_in = image.convert('RGB')
            rgb_in.save(dest_path, 'JPEG')


def show_history(history):
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


def prediction_model(list_class_):
    model = load_model('Models.h5')
    image_path = 'jakarta.jpg'

    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img) / 255
    img = np.expand_dims(img, axis=0)

    predict = model.predict(img)
    predict = np.argmax(predict)
    predict = list_class_[predict]
    print(predict)

    show_img = Image.open(image_path)
    plt.imshow(show_img)
    plt.show()


def training(train_, validation_):
    active = 'relu'
    # model = Sequential([
    #     Conv2D(128, activation=active, strides=1, kernel_size=3, input_shape=(150, 150, 3)),
    #     MaxPooling2D(2, 2),
    #
    #     Conv2D(64, activation=active, strides=1, kernel_size=3,),
    #     BatchNormalization(renorm=True),
    #     MaxPooling2D(2, 2),
    #
    #     Conv2D(64, activation=active, strides=1, kernel_size=3,),
    #     BatchNormalization(renorm=True),
    #     MaxPooling2D(2, 2),
    #
    #     Flatten(),
    #     Dense(256, activation=active),
    #     Dense(15, activation='softmax')
    # ])

    base_model = VGG19(
        pooling='avg',
        include_top=False,
        weights='imagenet',
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    model = Sequential([
        base_model,
        BatchNormalization(renorm=True),
        Flatten(),
        Dense(512, activation=active, activity_regularizer=regularizers.L1L2(0.02)),
        Dense(15, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(),
        loss=sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    model.summary()
    history = model.fit(
        train_,
        epochs=20,
        validation_data=validation_,
        batch_size=4
    )
    show_history(history)
    model.save('Models.h5')


def prepare_data(data_):
    date_iterator = data_.as_numpy_iterator()
    batch = date_iterator.next()

    ## check images
    # fig, ax = plt.subplots(ncols=4, figsize=(30, 30))
    # for i, img in enumerate(batch[0][:4]):
    #     ax[i].imshow(img.astype(int))
    #     ax[i].title.set_text(batch[1][i])
    #
    # plt.show()

    scaled_data = data_.map(lambda x, y: (x/255, y))

    ## check images after scale
    # fig, ax = plt.subplots(ncols=4, figsize=(30, 30))
    # for i, img in enumerate(scaled_data.as_numpy_iterator().next()[0][:4]):
    #     ax[i].imshow(img)
    #     ax[i].title.set_text(batch[1][i])
    #
    # plt.show()
    train_size = int(len(scaled_data)*.8)
    val_size = int(len(scaled_data)*.2)+1

    print(f'batch_size = {len(scaled_data)}')
    print(f'batch_size train = {train_size}')
    print(f'batch_size val   = {val_size}')

    train = scaled_data.take(train_size)
    validation = scaled_data.skip(train_size).take(val_size)
    training(train, validation)

if __name__ == '__main__':
    category = [
        'asinan-jakarta',
        'ayam-betutu',
        'bika-ambon',
        'bubur-manado',
        'es-dawet',
        'gado-gado',
        'gudeg',
        'gulai-ikan-mas',
        'kerak-telor',
        'mie-aceh',
        'nasi-goreng-kampung',
        'rawon',
        'rendang',
        'sate',
        'soto',
    ]

    dir_ = 'food-tfk-images/data/'
    ## change all image to jpeg
    # make_to_jpeg(dir_)

    data = image_dataset_from_directory(dir_, batch_size=4, image_size=(150, 150), shuffle=True)
    list_class = data.class_names
    prepare_data(data)
    # prediction_model(list_class)