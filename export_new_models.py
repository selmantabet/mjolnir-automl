from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2, InceptionV3, VGG16, VGG19, Xception, MobileNetV2, DenseNet121, EfficientNetV2S  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from keras.metrics import Precision, Recall, AUC
import os
from custom_metrics import f1_score
from wildfirenet import *

callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(filepath=os.path.join("tmp", 'temp_model.keras'),
                    monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

metrics_list = ['accuracy',
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc"),
                f1_score
                ]


def generate_model(bm):
    if bm == CUSTOMNET:
        model = bm
        model.compile(optimizer='adam',
                      loss='binary_crossentropy', metrics=metrics_list)
        model.save(os.path.join("tmp", model.name + ".keras"))
        return

    base_model = bm(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    # Create the model
    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name=bm.__name__)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=metrics_list)
    model.save(os.path.join("tmp", model.name + ".keras"))


base_models = [ResNet50V2, InceptionV3, VGG16, VGG19, Xception,
               MobileNetV2, DenseNet121, EfficientNetV2S, CUSTOMNET]
for bm in base_models:
    generate_model(bm)
