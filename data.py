import numpy as np
import os
import random
import imageio
import tensorflow as tf
from scipy import misc


def image_file_to_array(filename: str, dim_input: int):
    image = imageio.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255
    image = 1.0 - image

    return image

def get_images(paths, labels, n_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if n_samples is not None:
        sampler = lambda x: random.sample(x, n_samples)
    else:
        sampler = lambda x: x

    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


class DataGenerator(object):

    def __init__(self, num_classes: int,
                 num_samples_per_class: int,
                 num_meta_test_classes: int,
                 num_meta_test_samples_per_class: int,
                 config={}):

        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
        self.num_meta_test_classes = num_meta_test_classes
        data_folder = config.get("data_folder", "../omniglot_resized")
        self.img_size = config.get("image_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = num_classes

        class_folders = [
            os.path.join(data_folder, family, char)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for char in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, char))
        ]

        random.shuffle(class_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = class_folders[: num_train]
        self.metaval_character_folders = class_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = class_folders[
            num_train + num_val:]


    def sample_batch(self, batch_type, batch_size, shuffle=True, swap=False):
        """
        Samples a batch for training, validation, or testing
        Args:
        batch_type: meta_train/meta_val/meta_test
        shuffle: randomly shuffle classes or not
        swap: swap number of classes (N) and number of samples per class (K) or not
        Returns:
        A a tuple of (1) Image batch and (2) Label batch where
        image batch has shape [B, N, K, 784] and label batch has shape [B, N, K, N] if swap is False
        where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "meta_train":
            folders = self.metatrain_character_folders
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        elif batch_type == "meta_val":
            folders = self.metaval_character_folders
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        else:
            folders = self.metatest_character_folders
            num_classes = self.num_meta_test_classes
            num_samples_per_class = self.num_meta_test_samples_per_class
        
        all_image_batches, all_label_batches = [], []

        for _ in range(batch_size):
            sampled_character_folders = random.sample(folders, num_classes)
            labels_and_images = get_images(sampled_character_folders, range(
                num_classes), n_samples=num_samples_per_class, shuffle=False)
            labels = [li[0] for li in labels_and_images]
            images = [image_file_to_array(
                li[1], self.dim_input) for li in labels_and_images]
            images = np.stack(images)
            labels = np.array(labels).astype(np.int32)
            labels = np.reshape(
                labels, (num_classes, num_samples_per_class))
            labels = np.eye(num_classes, dtype=np.float32)[labels]
            images = np.reshape(
                images, (num_classes, num_samples_per_class, -1))

            batch = np.concatenate([labels, images], 2)
            if shuffle:
                for p in range(num_samples_per_class):
                    np.random.shuffle(batch[:, p])

            labels = batch[:, :, :num_classes]
            images = batch[:, :, num_classes:]

            if swap:
                labels = np.swapaxes(labels, 0, 1)
                images = np.swapaxes(images, 0, 1)

            all_image_batches.append(images)
            all_label_batches.append(labels)
    
        all_image_batches = np.stack(all_image_batches)
        all_label_batches = np.stack(all_label_batches)
        
        return all_image_batches, all_label_batches


if __name__ == "__main__":
    print("hi!")
    A = DataGenerator(1, 1, 1, 1)
