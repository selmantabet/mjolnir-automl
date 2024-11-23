import os
DATASETS = {
    "dataset_1": {
        "train": os.path.join("dataset_1", "train"),
        "test": os.path.join("dataset_1", "test"),
        "val": os.path.join("dataset_1", "val"),
        "augment": True
    },
    "dataset_2": {
        "train": os.path.join("dataset_2", "Training and Validation"),
        "test": os.path.join("dataset_2", "Testing"),
        "augment": True
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
