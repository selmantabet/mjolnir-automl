from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
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


def val_split(dataset, samples=None, val_size=0.2):
    # Calculate the number of samples for validation and training
    val_size = int(val_size * samples)
    train_size = samples - val_size
    # Print the sizes of the datasets
    print("Splitted dataset:")
    print(f"Training dataset size: {train_size} samples")
    print(f"Validation dataset size: {val_size} samples")
    val_dataset = dataset.take(val_size)
    train_dataset = dataset.skip(val_size)
    return train_dataset, val_dataset, train_size, val_size


def create_generators(directory, batch_size=32, img_height=224, img_width=224, augment=True, shuffle=True):
    generator = original_datagen.flow_from_directory(
        directory,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        shuffle=shuffle
    )
    if augment:
        augmented_generator = augmented_datagen.flow_from_directory(
            directory,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=shuffle
        )
        return generator, augmented_generator
    return generator, None


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


def class_counts_from_generators(generator, augmented_generator=None):
    print("--------------------")
    print("Number of samples in generator:", generator.samples)
    print("Number of classes:", generator.num_classes)
    print("--------------------")
    class_indices = generator.class_indices
    print("Class indices:", class_indices)
    class_names = list(class_indices.keys())
    print("Class names:", class_names)

    original_class_counts = {class_name: 0 for class_name in class_names}
    augmented_class_counts = {class_name: 0 for class_name in class_names}

    for class_name, class_index in class_indices.items():
        original_class_counts[class_name] = sum(
            generator.classes == class_index)
        if augmented_generator is not None:
            augmented_class_counts[class_name] = sum(
                augmented_generator.classes == class_index)

    # Print the results
    print("Dataset Class Counts:")
    for class_name, count in original_class_counts.items():
        print(f"{class_name}: {count}")
    if augmented_generator is not None:
        print("\nAugmented Dataset Class Counts:")
        for class_name, count in augmented_class_counts.items():
            print(f"{class_name}: {count}")
        print("\n")
        print("Combined Dataset Class Counts:")
        for class_name, count in augmented_class_counts.items():
            print(f"{class_name}: {count + original_class_counts[class_name]}")
    print("--------------------")

    combined_class_counts = original_class_counts.copy()
    for key in original_class_counts:
        combined_class_counts[key] += augmented_class_counts[key]

    return combined_class_counts


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
