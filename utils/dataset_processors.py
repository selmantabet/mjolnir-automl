""" 
The Data Processing Module - Part of the `utils` library for Project Mjölnir

Developed by Selman Tabet @ https://selman.io/
----------------------------------------------
This module contains functions for various operations on `keras.preprocessing.image.ImageDataGenerator` objects and other dataset-related tasks.
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
# Switch to a non-interactive backend to allow for cmd line execution
plt.switch_backend('agg')

# ImageDataGenerator for original images (no augmentation)
original_datagen = ImageDataGenerator(rescale=1./255)

# ImageDataGenerator for augmented images
augmented_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def reset_all_generators(generators):
    """
    Resets all generators in the provided list.
    This function iterates through a list of `generator` objects and calls the 
    `reset` method on each generator that is not `None`. It then returns the 
    modified list of `generators`.

    Arguments:
    -----------
        generators (`list[ImageDataGenerator]`): A list of generator objects, some of which may be None.

    Returns:
    --------
        `list`: The list of generators after reset.
    """

    for generator in generators:
        if generator is not None:
            generator.reset()
    return generators


def create_split_datagen(val_size=0.2):
    """
    Creates and returns two `ImageDataGenerator` instances for the original and augmented datasets.

    Arguments:
    -----------
        val_size (`float`): The proportion of the dataset to be used for validation. Default is 0.2.

    Returns:
    --------
        `tuple`: A tuple containing two `ImageDataGenerator` instances:
            - original_datagen: `ImageDataGenerator` for the original dataset with rescaling and validation split.
            - augmented_datagen: `ImageDataGenerator` for the augmented dataset with rescaling, various augmentations, and validation split.
    """

    # Create the ImageDataGenerator for the original dataset
    original_datagen = ImageDataGenerator(
        rescale=1./255, validation_split=val_size)
    # Create the ImageDataGenerator for the augmented dataset
    augmented_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=val_size
    )
    return original_datagen, augmented_datagen


def samples_from_generators(generators):
    """
    Calculate the total number of samples from a list of generators.

    Arguments:
    -----------
        generators (`list[ImageDataGenerator]`) : A list of generator objects.

    Returns:
    ---------
        `int`: The total number of samples from all generators in the list.
    """

    samples = 0
    for generator in generators:
        if generator is not None:
            samples += generator.samples
    return samples


def create_dataset(generator, batch_size=32, img_height=224, img_width=224):
    """
    Creates a TensorFlow dataset from a generator, applying batching and prefetching in the process.

    Arguments:
    -----------
        generator (`ImageDataGenerator`): A generator that yields tuples of `(image, label)`.
        batch_size (`int`, optional): The size of the batches to be produced. Defaults to `32`.
        img_height (`int`, optional): The height of the images. Defaults to `224`.
        img_width (`int`, optional): The width of the images. Defaults to `224`.

    Returns:
    --------
        `tf.data.Dataset`: A TensorFlow `DatasetV2` object.
    """

    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, img_height, img_width, 3], [None])
    )
    dataset = dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_generator(directory, batch_size=32, img_height=224, img_width=224, augment=True, shuffle=True):
    """
    Creates an image data generator for training or validation.

    Arguments:
    -----------
        directory (`str`): Path to the directory containing the image data.
        batch_size (`int`, optional): Number of images to return in each batch. Default is `32`.
        img_height (`int`, optional): Height of the images to be resized to. Default is `224`.
        img_width (`int`, optional): Width of the images to be resized to. Default is `224`.
        augment (`bool`, optional): Whether to apply data augmentation. Default is `True`.
        shuffle (`bool`, optional): Whether to shuffle the data. Default is `True`.

    Returns:
    --------
        `ImageDataGenerator`: An iterator yielding tuples of `(x, y)` where `x` is a numpy array of image data and `y` is a numpy array of corresponding labels.
    """

    if augment:
        generator = augmented_datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=shuffle
        )
        return generator
    else:
        generator = original_datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=shuffle
        )
        return generator


def class_weights_from_counts(class_counts, class_indices):
    """
    Calculate class weights from class counts.
    This function computes the weights for each class based on the provided 
    class counts. The weights are calculated as the total number of samples 
    divided by the product of the number of classes and the count of samples 
    in each class. The resulting weights are then mapped to their respective 
    class indices.

    Arguments:
    -----------
        class_counts (`dict`): A dictionary where keys are class names and values 
                             are the counts of samples in each class.
        class_indices (`dict`): A dictionary where keys are class names and values 
                              are the corresponding class indices.

    Returns:
    --------
        `dict`: A dictionary where keys are class indices and values are the 
              computed weights for each class.
    """

    total = sum(class_counts.values())
    weights = {class_name: total / (len(class_counts) * count)
               for class_name, count in class_counts.items()}
    weights = {class_indices[class_name]: weight
               for class_name, weight in weights.items()}
    return weights


def create_split_generators(directory, val_size=0.2, batch_size=32, img_height=224, img_width=224, augment=True, shuffle=True):
    """
    Creates training and validation data generators from a directory of images.

    Arguments:
    -----------
        directory (`str`): Path to the directory containing the image data.
        val_size (`float`, optional): Fraction of the data to be used for validation. Defaults to `0.2`.
        batch_size (`int`, optional): Number of images to be yielded from the generator per batch. Defaults to `32`.
        img_height (`int`, optional): Height of the input images. Defaults to `224`.
        img_width (`int`, optional): Width of the input images. Defaults to `224`.
        augment (`bool`, optional): Whether to apply data augmentation. Defaults to `True`.
        shuffle (`bool`, optional): Whether to shuffle the data. Defaults to `True`.

    Returns:
    --------
        `tuple`: A `tuple` containing two generators, one for training and one for validation.
    """

    original_datagen, augmented_datagen = create_split_datagen(val_size)

    def generator_from_datagen(datagen):
        """
        Create a generator from an ImageDataGenerator.

        Arguments:
        -----------
            datagen (`ImageDataGenerator`): An `ImageDataGenerator` object.

        Returns:
        --------
            `tuple`: A `tuple` containing two generators, one for training and one for validation.
        """
        original = datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=shuffle,
            subset='training'
        )  # Validation will never be augmented
        val = original_datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=False,
            subset='validation'
        )
        return original, val

    if augment:
        return generator_from_datagen(augmented_datagen)
    else:
        return generator_from_datagen(original_datagen)


def generators_to_dataset(generators, batch_size=32, img_height=224, img_width=224):
    """
    Converts a list of `ImageDataGenerator` into a single concatenated dataset.

    Arguments:
    -----------
        generators (`list`): A list of `ImageDataGenerator` objects.
        batch_size (`int`, optional): The size of the batches of data. Defaults to `32`.
        img_height (`int`, optional): The height of the images in the dataset. Defaults to `224`.
        img_width (`int`, optional): The width of the images in the dataset. Defaults to `224`.

    Returns:
    --------
        `tf.data.Dataset`: A concatenated `DatasetV2` object containing data from all the provided generators.
    """

    dataset = None
    for generator in generators:
        if generator is not None:
            if dataset is None:
                dataset = create_dataset(
                    generator, batch_size, img_height, img_width)
            else:
                dataset = dataset.concatenate(
                    create_dataset(generator, batch_size, img_height, img_width))
    return dataset


def class_counts_from_generator(generator):
    """
    This function takes a data generator, then calculates and returns a `dict` with the number of
    samples for each class in the dataset. It prints detailed information
    about the generator and the class distribution.

    Arguments:
    ----------
        generator (`ImageDataGenerator`): An `ImageDataGenerator` object.

    Returns:
    --------
        `dict`: A dictionary where keys are class names and values are the
              corresponding counts of samples in each class.
    """

    print("--------------------")
    print("Number of samples in generator:", generator.samples)
    print("Number of classes:", generator.num_classes)
    print("--------------------")
    class_indices = generator.class_indices
    print("Class indices:", class_indices)
    class_names = list(class_indices.keys())
    print("Class names:", class_names)

    class_counts = {class_name: 0 for class_name in class_names}

    for class_name, class_index in class_indices.items():
        class_counts[class_name] = sum(
            generator.classes == class_index)

    # Print the results
    print("Dataset Class Counts:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
    print("--------------------")

    return class_counts
