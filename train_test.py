import os
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.applications import *
from custom_metrics import f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from wildfirenet import create_wildfire_model

DATASETS = {
    # "The Wildfire Dataset": {
    #     "train": os.path.join("datasets", "dataset_1", "train"),
    #     "test": os.path.join("datasets", "dataset_1", "test"),
    #     "val": os.path.join("datasets", "dataset_1", "val"),
    #     # "augment": False,
    #     "source_url": "https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset/"
    # },
    # "DeepFire": {
    #     "train": os.path.join("datasets", "dataset_2", "Training"),
    #     "test": os.path.join("datasets", "dataset_2", "Testing"),
    #     # "augment": False,
    #     "source_url": "https://www.kaggle.com/datasets/alik05/forest-fire-dataset/"
    # },
    # "FIRE": {
    #     "train": os.path.join("datasets", "dataset_3"),
    #     # "augment": False,
    #     "source_url": "https://www.kaggle.com/datasets/phylake1337/fire-dataset/"
    # },
    "Forest Fire": {
        "train": os.path.join("datasets", "dataset_4", "train"),
        "test": os.path.join("datasets", "dataset_4", "test"),
        # "augment": False,
        "source_url": "https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images"
    },
}

cfg = {
    "train": DATASETS,
    # "test": "test_combined",
    "keras_models": [MobileNetV3Small, ResNet50V2, VGG19, DenseNet121],
    "custom_models": [create_wildfire_model(224, 224)],
    "hyperparameters": {
        "batch_size": 32,
        "epochs": 70,
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
    # "enforce_image_settings": True
}

# base_models = [ResNet50V2, InceptionV3, VGG16, VGG19, Xception, MobileNetV3Small, DenseNet121, EfficientNetV2S, CUSTOMNET]
