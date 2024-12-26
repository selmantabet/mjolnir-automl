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
    for generator in generators:
        if generator is not None:
            generator.reset()
    return generators


def create_split_datagen(val_size=0.2):
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
    samples = 0
    for generator in generators:
        if generator is not None:
            samples += generator.samples
    return samples

# Function to create a tf.data.Dataset from ImageDataGenerator


def create_dataset(generator, batch_size=32, img_height=224, img_width=224):
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, img_height, img_width, 3], [None])
    )
    dataset = dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_generator(directory, batch_size=32, img_height=224, img_width=224, augment=True, shuffle=True):
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
    total = sum(class_counts.values())
    weights = {class_name: total / (len(class_counts) * count)
               for class_name, count in class_counts.items()}
    weights = {class_indices[class_name]: weight
               for class_name, weight in weights.items()}
    return weights


def create_split_generators(directory, val_size=0.2, batch_size=32, img_height=224, img_width=224, augment=True, shuffle=True):
    original_datagen, augmented_datagen = create_split_datagen(val_size)

    def generator_from_datagen(datagen):
        original = datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=shuffle,
            subset='training'
        )
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


def consolidate_to_train(datasets):
    if len(datasets) == 1:
        return datasets[0]
    elif len(datasets) == 0:
        return None
    else:
        train_dataset = datasets[0]
        for dataset in datasets[1:]:
            train_dataset = train_dataset.concatenate(dataset)
        return train_dataset
