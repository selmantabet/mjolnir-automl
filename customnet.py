from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)


def build_customnet(img_height=224, img_width=224):
    customnet = Sequential()
    customnet.add(Conv2D(32, (3, 3), activation='relu',
                         input_shape=(img_height, img_width, 3), padding='same'))
    customnet.add(BatchNormalization())
    customnet.add(MaxPooling2D(pool_size=(2, 2)))
    customnet.add(Dropout(0.25))

    customnet.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    customnet.add(BatchNormalization())
    customnet.add(MaxPooling2D(pool_size=(2, 2)))
    customnet.add(Dropout(0.25))

    customnet.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    customnet.add(BatchNormalization())
    customnet.add(MaxPooling2D(pool_size=(2, 2)))
    customnet.add(Dropout(0.25))

    customnet.add(Flatten())
    customnet.add(Dense(256, activation='relu'))
    customnet.add(BatchNormalization())
    customnet.add(Dropout(0.5))
    customnet.add(Dense(128, activation='relu'))
    customnet.add(BatchNormalization())
    customnet.add(Dropout(0.5))

    customnet.add(Dense(1, activation='sigmoid'))
    customnet.name = 'WildfireNet'
    return customnet


CUSTOMNET = build_customnet()
