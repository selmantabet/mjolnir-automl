import os
DATASETS = {
    "The Wildfire Dataset": {
        "train": os.path.join("dataset_1", "train"),
        "test": os.path.join("dataset_1", "test"),
        "val": os.path.join("dataset_1", "val"),
        "augment": False,
        "source_url": "https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset/"
    },
    "DeepFire": {
        "train": os.path.join("dataset_2", "Training"),
        "test": os.path.join("dataset_2", "Testing"),
        "augment": False,
        "source_url": "https://www.kaggle.com/datasets/alik05/forest-fire-dataset/"
    },
    "FIRE": {
        "train": "dataset_3",
        "augment": False,
        "source_url": "https://www.kaggle.com/datasets/phylake1337/fire-dataset/"
    },
}
train_dirs = [DATASETS[dataset].get('train') for dataset in DATASETS]
test_dirs = [DATASETS[dataset].get('test') for dataset in DATASETS]
val_dirs = [DATASETS[dataset].get('val') for dataset in DATASETS]

print(DATASETS)
print(train_dirs)
print(test_dirs)
print(val_dirs)
data_set = list(zip(train_dirs, test_dirs, val_dirs))
print(data_set)
