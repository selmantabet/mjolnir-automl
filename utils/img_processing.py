""" 
The Image Processing Module - Part of the `utils` library for Project Mjölnir

Developed by Selman Tabet @ https://selman.io/
----------------------------------------------
This module contains functions for enforcing image parameters and plotting images.
"""
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
# Switch to a non-interactive backend to allow for command-line execution
plt.switch_backend('agg')
Image.MAX_IMAGE_PIXELS = None


def enforce_image_params(root_dir, target_size=(224, 224), quality=90):
    """
    Enforce image parameters for all images in the specified root directory, which is normally a dataset.

    This function will walk through all subdirectories of the root directory,
    resize images to the `target_size`, convert them to `RGB` mode if necessary,
    and overwrites them with the specified `quality`.

    Arguments:
    ----------
        root_dir (`str`): The root directory containing images.
        target_size (`tuple`): The target size for resizing images (default is `(224, 224)`).
        quality (`int`): The quality for saving images (default is `90`).

    Returns:
    --------
        `None`
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(subdir, file)
                with Image.open(file_path) as img:
                    if img.size == target_size and img.mode == 'RGB':
                        continue
                    if img.size != target_size:
                        img = img.resize(target_size, Image.LANCZOS)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(file_path, quality=quality, optimize=True)


def plot_images(directory, category, num_images, img_height=224, img_width=224):
    """
    Plots a specified number of images from a given category within a directory.

    Arguments:
    ----------
        directory (`str`): The path to the main directory containing image categories.
        category (`str`): The specific category of images to plot.
        num_images (`int`): The number of images to display.
        img_height (`int`, optional): The height to resize images to. Default is `224`.
        img_width (`int`, optional): The width to resize images to. Default is `224`.

    Returns:
    --------
        `None`
    """

    category_dir = os.path.join(directory, category)
    images = os.listdir(category_dir)[:num_images]

    plt.figure(figsize=(15, 5))
    dataset_name = os.path.basename(os.path.dirname(directory))
    plt.suptitle(f"Dataset: {dataset_name} - Category: {category}", y=0.8)
    for i, img_name in enumerate(images):
        img_path = os.path.join(category_dir, img_name)
        img = load_img(img_path, target_size=(img_height, img_width))
        img_array = img_to_array(img) / 255.0

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img_array)
        plt.title(category)
        plt.axis('off')
