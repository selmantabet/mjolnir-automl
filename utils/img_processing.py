import os
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
# Switch to a non-interactive backend to allow for cmd line execution
plt.switch_backend('agg')


def enforce_image_params(root_dir, target_size=(224, 224), quality=90):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(subdir, file)
                with Image.open(file_path).convert('RGB') as img:
                    if img.size != target_size:
                        img = img.resize(target_size, Image.LANCZOS)
                        img.save(file_path, quality=quality, optimize=True)


def plot_images(directory, category, num_images, img_height=224, img_width=224):
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
    # plt.show()
