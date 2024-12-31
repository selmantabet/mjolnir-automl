# %% [markdown]
# # Project Mjölnir: An Automated Brute-Force Dataset-Model Combinatorics Training and Evaluation Pipeline for Computer Vision - by Selman Tabet @ https://selman.io/

# %% [markdown]
# ### Importing Libraries

# %%
# Data processing libraries
import numpy as np
from itertools import combinations  # For defining search space
import json  # For saving and loading training results
import argparse  # For command line arguments
import os
import time

# Tensorflow-Keras ML helpers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model  # To plot model architecture

from IPython import get_ipython  # To check if code is running in Jupyter notebook
import importlib.util  # To import config module from str
from pprint import pprint  # To show config

# Custom helper functions
from utils.img_processors import enforce_image_params
# Dataset and generator processing functions
from utils.dataset_processors import *
from utils.plot_functions import *  # Plotting functions
from utils.evaluator import *  # Complete evaluation program
from utils.initializer import *  # Set temp path and other initializations
from utils.cfg_validator import *  # Validate config object

# %% [markdown]
# ## Specify pipeline parameters here
# The configuration object is the most important part of the pipeline. It contains all the parameters that the pipeline will use to train and evaluate the models. The configuration object is a dictionary that MUST be named ***'default_cfg'***. The configuration object must contain the following:
# ```python
# {
#     'datasets': { ***(required)***
#         "dataset_name1": {
#             'train': path str to the training dataset ***(required)***,
#             'val': path str to the validation dataset (optional),
#             'test': path str to the test dataset (optional),
#             'augment': bool to enable data augmentation on the training dataset (optional, default=True),
#         },
#         "dataset_name2": {}, # Keep adding datasets as needed
#         "dataset_name3": {},
#         ...
#     },
#     'full_test': path str to the test dataset (optional, recommended),
#     'val_size': float of the validation dataset to split (optional, default=0.2),
#     'keras_models': [list of instances of the keras.applications base models]
#     'custom_models': [list of instances of custom models, to be compiled without modification],
#     'hyperparameters': { ***(required)***
#         'batch_size': int ***(required)***,
#         'epochs': int ***(required)***,
#     },
#     'optimizer': instance of keras.optimizers.Optimizer or str ***(required)***,
#     'loss': instance of keras.losses.Loss or str ***(required)***,
#     'image_width': int (optional, default=224),
#     'image_height': int (optional, default=224),
#     'metrics': [list of metric functions] ***(required)***,
#     'callbacks': [list of instances of keras.callbacks.Callback] (optional),
#     'enforce_image_params': bool to force RGB color mode and image sizes according to above specs (optional)
#     'metric_weights': { 'metric1': 1,
#                         'metric2': 1.5,
#                         ...
#                         # All metrics must be present in the list of metrics and match their names
#                     } (optional)
# }
#
# ```
# The "full_test" configuration key is optional and is used to specify a test dataset that will be used to evaluate all models after training. If the "full_test" key is not provided, the pipeline will take the "test" paths provided under each dataset, then combine them to form a consolidated test set to evaluate all models with.
#
# **Note: Ensure that the dataset classes are in separate folders and that the folder names are the class names. The pipeline will automatically detect the classes from the dataset paths.**

# %%
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV2, VGG19, ResNet50V2, Xception, DenseNet121
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
    "FIRE": {
        "train": os.path.join("datasets", "dataset_3"),
    },
}

default_cfg = {
    "datasets": DATASETS,  # The datasets to use
    # This overrides the test datasets stored under "datasets"
    "full_test": os.path.join("datasets", "d4_test"),
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
                          patience=3, verbose=1)
    ],
    # If True, the image sizes and RGB colour mode will be enforced on all images
    "enforce_image_settings": True,
    "metric_weights": {  # Weights for each metric for Weighted Sum of Metrics Calculation
        "accuracy": 1,
        "precision": 1,
        "recall": 1.3,
        "auc": 1.2,
        "f1_score": 1
    }
}

# %% [markdown]
# ### Parse arguments from command line and setup configuration

# %%


def in_notebook():
    """ 
    Detect if script is running in a Jupyter notebook
    Generated using GPT-4o. Prompt: "Detect if running in a Jupyter notebook"
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        else:
            return False  # Other type (terminal, etc.)
    except NameError:
        return False      # Probably standard Python interpreter


parser = argparse.ArgumentParser(
    description="Parse command line arguments")
parser.add_argument('--from-py-cfg', type=str,
                    help='Path to the config Python file')
if not in_notebook():
    args = parser.parse_args()
    config_file_path = args.from_py_cfg
    print(f"Python Config Path: {config_file_path}")
else:
    config_file_path = False

if config_file_path:  # Load config from Python file
    spec = importlib.util.spec_from_file_location(
        "config_module", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    if hasattr(config_module, 'cfg'):
        config = config_module.cfg
    else:
        raise AttributeError(
            "The provided config module does not have a 'cfg' attribute.")
    print("Loaded config from Python file:")
    pprint(config)
else:
    print("No Python config file specified, using default (notebook) config.")
    if 'default_cfg' in globals():
        config = default_cfg
    else:
        raise AttributeError(
            "The notebook does not have a 'default_cfg' variable")

# Validate the config
print("Validating config...")
validate_config(config)

# %% [markdown]
# ### Loading parameters

# %%
# Load parameters from config
training_datasets = config.get('datasets', {})
full_test_dir = config.get('full_test')
base_models = config.get('keras_models', [])
custom_models = config.get('custom_models', [])
metric_weights = config.get('metric_weights', {})
hyperparameters = config.get(
    'hyperparameters', {"epochs": 50, "batch_size": 32})
epochs = hyperparameters.get('epochs', 50)
batch_size = hyperparameters.get('batch_size', 32)
img_height = config.get('image_height', 224)
img_width = config.get('image_width', 224)
optimizer_fn = config.get('optimizer', 'adam')
loss_fn = config.get('loss', 'binary_crossentropy')
callbacks_list = config.get('callbacks', [])
metrics_list = config.get('metrics', ['accuracy'])
enforce_image_size = config.get('enforce_image_settings', False)
val_size = config.get('val_size', 0.2)

# Check if training datasets are defined
if training_datasets is None or len(training_datasets) == 0:
    raise ValueError("No train datasets defined in config.")
# Check if either base_models or custom_models are defined
if base_models is None or len(base_models) == 0:
    if custom_models is None or len(custom_models) == 0:
        raise ValueError("No models defined in config.")

# %% [markdown]
# ### Parsing parameters

# %%
train_dirs = [training_datasets[ds].get('train') for ds in training_datasets]
test_dirs = [training_datasets[ds].get('test') for ds in training_datasets]
val_dirs = [training_datasets[ds].get('val') for ds in training_datasets]

# Combine all directories for image params enforcement
all_dirs = train_dirs + test_dirs + val_dirs + [full_test_dir]
all_dirs = [d for d in all_dirs if d is not None]  # Remove None values

# Combine base_models and custom_models to be looped over in training
all_models = base_models + custom_models
# Create a list to keep track of which models are custom
is_custom_model = [False] * len(base_models) + [True] * len(custom_models)

# %% [markdown]
# ### Enforce defined resolution and colour mode

# %%
if enforce_image_size:  # Enforce image size and RGB colour mode for all images
    for directory in all_dirs:
        print(f"Adjusting image properties in {directory}")
        enforce_image_params(directory, target_size=(img_width, img_height))

# %% [markdown]
# ### Create training and validation generators

# %%
dataset_names = []  # [ "dataset_1", "dataset_2", ... ]
train_generators = []  # [ (dataset_1_train, dataset_2_train), ... ]
train_sizes = []  # [ (dataset_1_train_size, dataset_2_train_size), ... ]
val_generators = []  # [ (dataset_1_val, dataset_2_val), ... ]
val_sizes = []  # [ (dataset_1_val_size, dataset_2_val_size), ... ]
train_counts = []  # [ (dataset_1_train_counts, dataset_2_train_counts), ... ]
val_counts = []  # [ (dataset_1_val_counts, dataset_2_val_counts), ... ]

for d in training_datasets:
    print(f"Processing: {d}")
    train_dir = training_datasets[d].get('train')
    augment = training_datasets[d].get('augment', True)
    print("Augmenting" if augment else "Not augmenting", d)
    # Apply original and augmented data generators for training
    print("Creating generators for training")
    if "val" in training_datasets[d]:  # Use separate validation dataset
        train_generator = create_generator(
            train_dir, batch_size=batch_size, augment=augment, img_width=img_width, img_height=img_height)
        val_generator = create_generator(
            training_datasets[d]['val'], batch_size=batch_size, augment=False, shuffle=False, img_width=img_width, img_height=img_height)
    else:  # Split the training dataset into training and validation
        print(f"No validation set provided for {
              d}, splitting training dataset.")
        train_generator, val_generator = create_split_generators(
            train_dir, val_size=val_size, batch_size=batch_size, augment=augment, img_width=img_width, img_height=img_height)

    train_samples = train_generator.samples
    # Class indices must be consistent across training and validation, assertion will be made later
    class_indices = train_generator.class_indices
    train_count_dict = class_counts_from_generator(train_generator)

    val_samples = val_generator.samples
    val_count_dict = class_counts_from_generator(val_generator)

    # Calculate the number of samples for training and validation
    train_sizes.append(train_samples)
    val_sizes.append(val_samples)

    train_counts.append(train_count_dict)
    val_counts.append(val_count_dict)
    train_generators.append(train_generator)
    val_generators.append(val_generator)
    dataset_names.append(d)

# Ensure that the lengths are consistent across the board before continuing
assert len(train_sizes) == len(train_generators) == len(val_sizes) == len(val_generators) == len(
    val_counts) == len(train_counts) == len(dataset_names), "Dataset lengths are inconsistent."


# %% [markdown]
# ### Brute Force Combinatorial Search Space Definition

# %%
# [(0,), (1,), (0, 1), ...] where 0, 1 are the indices of the datasets within their respective lists
dataset_combos = []
for r in range(1, len(dataset_names) + 1):  # For all combination sizes
    # Generate all possible combinations of datasets for each size
    dataset_combos.extend(combinations(range(len(dataset_names)), r))

combined_training_datasets = []
combined_val_datasets = []
combined_dataset_names = []
steps_per_epoch_list = []
validation_steps_list = []
train_counts_list = []
val_counts_list = []

for combo in dataset_combos:
    # Initialize variables for each combination
    train_generators_list = None
    val_generators_list = None
    train_size = None
    val_size = None
    train_count = None
    val_count = None
    for idx in combo:
        if train_generators_list is None:
            # Initialize the lists with the first dataset
            train_generators_list = [train_generators[idx]]
            val_generators_list = [val_generators[idx]]
            train_size = train_sizes[idx]
            val_size = val_sizes[idx]
            train_count = train_counts[idx]
            val_count = val_counts[idx]
        else:
            # Combine the rest of the datasets
            train_generators_list.append(train_generators[idx])
            val_generators_list.append(val_generators[idx])
            train_size += train_sizes[idx]
            val_size += val_sizes[idx]
            train_count = {k: train_count.get(
                k, 0) + train_counts[idx].get(k, 0) for k in set(train_count) | set(train_counts[idx])}
            val_count = {k: val_count.get(
                k, 0) + val_counts[idx].get(k, 0) for k in set(val_count) | set(val_counts[idx])}
        # NumPy int64 to int cast
        train_count = {k: int(v) for k, v in train_count.items()}
        val_count = {k: int(v) for k, v in val_count.items()}

    # Combine all accumulated generators for this combination into a single dataset
    training_dataset = generators_to_dataset(
        train_generators_list, batch_size=batch_size, img_height=img_height, img_width=img_width)
    val_dataset = generators_to_dataset(
        val_generators_list, batch_size=batch_size, img_height=img_height, img_width=img_width)
    # Append the combined datasets and other relevant parameters to their respective lists
    combined_dataset_names.append(
        "_".join([dataset_names[idx] for idx in combo]))
    combined_training_datasets.append(training_dataset)
    combined_val_datasets.append(val_dataset)
    steps_per_epoch_list.append(train_size // batch_size)
    validation_steps_list.append(val_size // batch_size)
    train_counts_list.append(train_count)
    val_counts_list.append(val_count)
    # Zip all the lists together for easier unpacking in the training loop
    training_params = list(zip(combined_dataset_names, combined_training_datasets, combined_val_datasets,
                           steps_per_epoch_list, validation_steps_list, train_counts_list, val_counts_list))

# %% [markdown]
# ### Generate the test dataset

# %%
if full_test_dir is None:
    test_generators = []
    print("No target test directory provided, merging all tests from provided datasets if available.")
    for d in test_dirs:
        if d is not None:
            test_generators.append(create_generator(d, batch_size=batch_size, augment=False, shuffle=False,
                                   # No augmentation/shuffle for testing
                                                    img_height=img_height, img_width=img_width))
    if len(test_generators) == 0:
        raise ValueError("No tests found in the provided datasets.")

    test_steps = sum([gen.samples for gen in test_generators]) // batch_size
    test_dataset = generators_to_dataset(
        test_generators, batch_size=batch_size, img_height=img_height, img_width=img_width)
else:
    test_generators = [create_generator(full_test_dir, batch_size=batch_size, augment=False, shuffle=False,
                                        # No augmentation/shuffle for testing
                                        img_height=img_height, img_width=img_width)]
    test_steps = test_generators[0].samples // batch_size
    test_dataset = create_dataset(
        test_generators[0], batch_size=batch_size, img_height=img_height, img_width=img_width)
if len(test_generators) > 0:
    assert test_generators[0].class_indices == train_generators[0].class_indices, "Test and training class indices do not match, check the provided directories and their class names."
else:
    raise ValueError("No test generators were created.")

print("Test Dataset Class Counts:")
for gen in test_generators:
    print("Class indices:", gen.class_indices)
    for class_name, class_index in gen.class_indices.items():
        print(f"{class_name}: {sum(gen.classes == class_index)}")
print("\n")

# %% [markdown]
# ### Model Preparation

# %%


def generate_model(bm, custom=False, to_dir=TEMP_DIR):
    if custom:  # Custom models are compiled and saved as is
        model = bm
        model.compile(optimizer=optimizer_fn,
                      loss=loss_fn, metrics=metrics_list)
        os.makedirs(os.path.join(to_dir, model.name), exist_ok=True)
        model.save_weights(os.path.join(to_dir, model.name, f"{
                           model.name}_initial.weights.h5"))
        return model

    base_model = bm(
        include_top=False,
        weights='imagenet',  # Use pre-trained weights for transfer learning
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False  # Freeze the base model weights for transfer learning

    # Create the model
    inputs = Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)  # Pooling layer for dimensionality reduction
    x = BatchNormalization()(x)  # Batch normalization layer for stability
    x = Dropout(0.5)(x)  # Regularization layer
    # Trainable layer for this application
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)  # Batch normalization layer
    x = Dropout(0.5)(x)  # Regularization layer
    outputs = Dense(1, activation='sigmoid')(x)  # Binary classification output

    model = Model(inputs, outputs, name=bm.__name__)
    model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=metrics_list)
    os.makedirs(os.path.join(to_dir, model.name), exist_ok=True)
    model.save_weights(os.path.join(to_dir, model.name, f"{
                       model.name}_initial.weights.h5"))
    return model

# %% [markdown]
# ### Training and evaluating the models and combinations


# %%
if not os.path.exists("runs"):
    os.makedirs("runs")

run_number = len([d for d in os.listdir("runs") if os.path.isdir(
    os.path.join("runs", d)) and d.startswith('run_')]) + 1
run_dir = os.path.join("runs", f"run_{run_number}")
os.makedirs(run_dir, exist_ok=True)

# %%
run_config = {  # Save basic run info for reference
    "datasets": training_datasets,
    "val_size": val_size,
    "hyperparameters": hyperparameters,
    "test_dirs": test_dirs,
    "full_test": full_test_dir,
    "number_of_models": len(all_models),
    "metric_weights": metric_weights,
}

with open(os.path.join(run_dir, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=4)

# %%
training_results = {}
results_file = os.path.join(run_dir, 'training_results.json')

for base_model, custom_bool in zip(all_models, is_custom_model):
    # Generate the model and its initial weights
    model = generate_model(base_model, custom=custom_bool, to_dir=run_dir)
    model.summary()
    model_dir = os.path.join(run_dir, model.name)
    # Initialize the model results dictionary
    training_results[model.name] = {}
    plot_model(model, show_shapes=True, show_layer_names=True,
               to_file=os.path.join(model_dir, f"{model.name}_architecture.png"))
    # Main training and evaluation loop - unpack the training parameters for each combination
    for dataset_id, train_dataset, val_dataset, steps_per_epoch, validation_steps, train_counts_dict, val_counts_dict in training_params:
        # Reload the initial weights for each dataset
        model.load_weights(os.path.join(run_dir, model.name,
                           f"{model.name}_initial.weights.h5"))
        print(f"Training model: {model.name} on dataset: {dataset_id}")
        # Calculate class weights of the current dataset for class-weighted training
        class_weights = class_weights_from_counts(
            train_counts_dict, class_indices=class_indices)
        print("Class weights:", class_weights)
        # Record the start time
        start_time = time.time()

        # Model training
        history = model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            class_weight=class_weights
        )

        # Record the end time
        end_time = time.time()
        # Calculate the training time
        training_time = end_time - start_time
        print(f"Training time: {training_time:.2f} seconds")

        model_ds_dir = os.path.join(model_dir, dataset_id)
        os.makedirs(model_ds_dir, exist_ok=True)
        # Save the model
        model.save(os.path.join(model_ds_dir, f"{
                   model.name}_{dataset_id}.keras"))

        ###### Evaluation stage ######
        optimal_threshold = full_evaluation(
            model_ds_dir, history, model, dataset_id, test_generators)
        evaluation = model.evaluate(
            test_dataset, return_dict=True, steps=test_steps)

        training_results[model.name][dataset_id] = {
            'history': history.history,
            'training_time': training_time,
            'optimal_threshold': float(optimal_threshold),
            'train_dataset_size': steps_per_epoch * batch_size,
            'val_dataset_size': validation_steps * batch_size,
            'train_counts': train_counts_dict,
            'val_counts': val_counts_dict,
            'train_counts_total': sum(train_counts_dict.values()),
            'val_counts_total': sum(val_counts_dict.values()),
            # np.float64 to float typecast
            'class_weights': {k: float(v) for k, v in class_weights.items()},
            "evaluation": evaluation
        }
        print(f"Training results for {model.name} on {dataset_id}:")
        # Print the results for this model and dataset
        pprint(training_results[model.name][dataset_id])
        # Save the training results to a file after each iteration
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=4)

        # Reset the model for the next iteration
        model.compile(optimizer=optimizer_fn,
                      loss=loss_fn, metrics=metrics_list)

# %%
print("Brute force loop completed!")
print(f"All models are now available at: {run_dir}")

# %% [markdown]
# ### Aggregation Stage

# %%
eval_dir = os.path.join(run_dir, "evaluations")
os.makedirs(eval_dir, exist_ok=True)
# Extract the training and evaluation data from the training results and save it to a CSV file
rows = extract_evaluation_data(training_results)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(eval_dir, "training_data.csv"), index=False)

# %%
plot_metric_chart(df, "Training Time", eval_dir, highlight_max=False)
plot_dataset_sizes(df, eval_dir)

for metric in evaluation:
    if metric == "loss":
        continue  # Loss is not useful in this context
    else:
        plot_metric_chart(df, metric, eval_dir)

plot_time_extrapolation(df, eval_dir)
plot_sum_of_metrics_heatmaps(eval_dir, df, metric_weights)
print("All evaluations completed!")
print(f"Results are available at: {eval_dir}")
