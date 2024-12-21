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
        # "augment": False,
        "source_url": "https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset/"
    },
    "DeepFire": {
        "train": os.path.join("datasets", "dataset_2", "Training"),
        "test": os.path.join("datasets", "dataset_2", "Testing"),
        # "augment": False,
        "source_url": "https://www.kaggle.com/datasets/alik05/forest-fire-dataset/"
    },
    "FIRE": {
        "train": os.path.join("datasets", "dataset_3"),
        # "augment": False,
        "source_url": "https://www.kaggle.com/datasets/phylake1337/fire-dataset/"
    },
    # "Forest Fire": {
    #     "train": os.path.join("datasets", "dataset_4", "train"),
    #     "test": os.path.join("datasets", "dataset_4", "test"),
    #     # "augment": False,
    #     "source_url": "https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images"
    # },
}

default_cfg = {
    "datasets": DATASETS,  # The datasets to use
    # This overrides the test datasets stored under "datasets"
    "test": os.path.join("datasets", "d4_test"),
    "val_size": 0.2,  # The size of the validation dataset if splitting is needed
    "keras_models": [MobileNetV3Small, MobileNetV2, VGG19, ResNet50V2, Xception, DenseNet121],
    "custom_models": [create_wildfire_model(224, 224)],  # Custom models to use
    "hyperparameters": {
        "batch_size": 32,
        "epochs": 80,
    },
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "image_width": 224,
    "image_height": 224,
    "metrics": ['accuracy',  # Metrics functions, directly handed to model.compile
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc"),
                f1_score
                ],
    "callbacks": [  # Callback functions, directly handed to model.fit
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join("tmp", 'temp_model.keras'),
                        monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, verbose=1)
    ],
    # If True, the image sizes and RGB colour mode will be enforced on all images
    "enforce_image_settings": True
}

# base_models = [ResNet50V2, InceptionV3, VGG16, VGG19, Xception, MobileNetV3Small, DenseNet121, EfficientNetV2S, CUSTOMNET]
