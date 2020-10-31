import numpy as np
import os, random, imageio
import tensorflow as tf
from scipy import misc

def image_file_to_array(filename: str, dim_input: int):
    image = imageio.read(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255
    image = 1.0 - image

    return image


class DataGenerator(object):

    def __init__(self, num_classes: int,
                 num_samples_per_class: int,
                 num_meta_test_classes: int,
                 num_meta_test_samples_per_class: int,
                 config={}):

        
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
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


if __name__ == "__main__":
    print("hi!")
    A = DataGenerator(1, 1, 1, 1)

