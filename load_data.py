import numpy as np
import os
import random
#import tensorflow as tf
from scipy import misc


def get_images(paths, labels, nb_samples=None, shuffle=True):
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
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, dim_input):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = misc.imread(filename)
    image = image.reshape([dim_input])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', './omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        # Per batch 1
        
        #set_trace()
        batch_response      =   []
        batch_lbl_response  =   []
        for _ibatch in range(batch_size):
            paths   =   random.sample(folders, self.num_classes)
            labels  =   [np.zeros(self.num_classes) for _ in range(self.num_classes)]
            for i in range(len(labels)):
                labels[i][i] = 1.0
            _images_links = get_images(paths, labels, nb_samples=self.num_samples_per_class, shuffle=False)
            _imgs   =   [image_file_to_array(filename, -1) for _, filename in _images_links]
            _imgs_stack     =   [_imgs[idx:idx+self.num_samples_per_class] for idx in range(0, len(_imgs), self.num_samples_per_class)]
            _imgs_stack     =   np.dstack(_imgs_stack)
            _imgs_stack     =   np.transpose(_imgs_stack, (0, 2, 1))
            _labels =   [_lab for _lab, _ in _images_links]
            _labels_stack   =   [_labels[idx:idx+self.num_samples_per_class] for idx in range(0, len(_labels), self.num_samples_per_class)]
            _labels_stack   =   np.dstack(_labels_stack)
            _labels_stack   =   np.transpose(_labels_stack, (0, 2, 1))
            batch_response.append(np.expand_dims(_imgs_stack, 0))
            batch_lbl_response.append(np.expand_dims(_labels_stack, 0))
        
        all_image_batches   =   np.vstack(batch_response)
        all_label_batches   =   np.vstack(batch_lbl_response)
        #############################

        return all_image_batches, all_label_batches
