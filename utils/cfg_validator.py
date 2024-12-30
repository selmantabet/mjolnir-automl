"""
The Input Config Validation Script - Project Mjölnir

Developed by Selman Tabet @ https://selman.io/
----------------------------------------------
This script contains functions for validating the configuration objects.

Failed assertions will raise an AssertionError with a descriptive message.
"""
import os


def validate_config_structure(config):
    assert isinstance(config, dict), "Configuration must be a dictionary."
    required_keys = ['datasets', 'hyperparameters',
                     'metrics', 'optimizer', 'loss']
    for key in required_keys:
        assert key in config, f"Missing required key in config: {key}"


def validate_data_types(config):
    type_checks = {
        'enforce_image_settings': bool,
        'datasets': dict,
        'hyperparameters': dict,
        'metrics': list,
        'callbacks': list,
        'image_width': int,
        'image_height': int,
        'metric_weights': dict,
        'val_size': float
    }

    for key, expected_type in type_checks.items():
        if key in config:
            assert isinstance(config[key], expected_type), f"{key} must be a {
                expected_type.__name__}, got {type(config[key]).__name__}"


def validate_datasets_paths(config):
    for dataset_name, dataset_paths in config['datasets'].items():
        assert 'train' in dataset_paths, f"Missing train key for dataset {
            dataset_name}."
        for key, path in dataset_paths.items():
            assert os.path.exists(path), f"Path does not exist for {
                key} in dataset {dataset_name}: {path}"
    if "full_test" in config:
        assert os.path.exists(config['full_test']), f"Path does not exist for test dataset: {
            config['full_test']}"


def validate_dataset_loading(config):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    for dataset_name, dataset_paths in config['datasets'].items():
        train_gen = datagen.flow_from_directory(
            dataset_paths['train'],
            target_size=(config.get('image_width', 224),
                         config.get('image_height', 224)),
            batch_size=config['hyperparameters'].get('batch_size', 32),
            class_mode='binary'
        )
        assert train_gen.samples > 0, f"No samples found in training set for {
            dataset_name}"


def validate_model_compilation(config):
    if 'custom_models' in config:
        for custom_model in config['custom_models']:
            model = custom_model
            try:  # No straightforward to check model except by trying to compile it
                model.compile(
                    optimizer=config['optimizer'],
                    loss=config['loss'],
                    metrics=config['metrics']
                )
            except TypeError:
                raise AssertionError(
                    f"{model} does not support the compile method.")
            assert model.optimizer is not None, f"Optimizer is not set correctly for custom model {
                model.name}."
            assert model.loss == config['loss'], f"Loss is not set correctly for custom model {
                model.name}."
    if 'keras_models' in config:
        for base_model in config['keras_models']:
            try:
                model = base_model(
                    input_shape=(224, 224, 3), weights=None, classes=1)
                model.compile(
                    optimizer=config['optimizer'],
                    loss=config['loss'],
                    metrics=config['metrics']
                )
            except TypeError:
                raise AssertionError(
                    f"{base_model} does not support the compile method.")
            assert model.optimizer is not None, f"Optimizer is not set correctly for Keras model {
                model.name}."
            assert model.loss == config['loss'], f"Loss is not set correctly for Keras model {
                model.name}."


def validate_metrics(config):
    from tensorflow.keras.metrics import get as get_metric
    for metric in config['metrics']:
        try:
            get_metric(metric)
        except ValueError:
            raise AssertionError(
                f"Metric {metric} is not recognized by Keras.")


def validate_callbacks(config):
    from tensorflow.keras.callbacks import Callback
    for callback in config['callbacks']:
        assert isinstance(callback, Callback), f"Callback {
            callback} is not an instance of Keras Callback."


def validate_config(config):
    validate_config_structure(config)
    validate_data_types(config)
    validate_datasets_paths(config)
    validate_dataset_loading(config)
    validate_model_compilation(config)
    validate_metrics(config)
    validate_callbacks(config)
    print("Configuration validation successful.")
