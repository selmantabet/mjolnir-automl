""" 
Custom Configuration Script - Project Mjölnir

Developed by Selman Tabet @ https://selman.io/
----------------------------------------------
This script contains the configuration for the wildfire detection example.

The configuration file can be directly submitted to the main pipeline script with the --from-py-cfg flag.

Do not modify the variable name `cfg` as it is used by the main pipeline script.

Feel free to modify the configuration parameters as needed, make copies of this file for different experiments and use 
a shell script to run multiple experiments in one batch.
"""

import os
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.applications import *
from custom_metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# WildfireNet model, for comparison to other SOTA models in dissertation.
from wildfirenet import create_wildfire_model

DATASETS = {
    "The Wildfire Dataset": {
        "train": os.path.join("datasets", "dataset_1", "train"),
        "test": os.path.join("datasets", "dataset_1", "test"),
        "val": os.path.join("datasets", "dataset_1", "val"),
    },
    "DeepFire": {
        "train": os.path.join("datasets", "dataset_2", "Training"),
        "test": os.path.join("datasets", "dataset_2", "Testing"),
    },
}

cfg = {  # DO NOT MODIFY THE VARIABLE NAME
    "datasets": DATASETS,
    "full_test": "test_combined",
    "val_size": 0.2,  # The size of the validation dataset if splitting is needed
    "keras_models": [MobileNetV3Small, Xception, InceptionV3, ResNet50, VGG16],
    "custom_models": [create_wildfire_model(224, 224)],
    "hyperparameters": {
        "batch_size": 32,
        "epochs": 80,
    },
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "image_width": 224,
    "image_height": 224,
    "metrics": ['accuracy',
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc"),
                f1_score
                ],
    "callbacks": [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join("tmp", 'temp_model.keras'),
                        monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, verbose=1)
    ],
    "enforce_image_settings": True
}
