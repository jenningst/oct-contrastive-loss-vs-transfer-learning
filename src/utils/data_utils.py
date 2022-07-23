import tensorflow as tf
import os
import math
import matplotlib.pyplot as plt
      
def prepare_dataset(unlabeled_dir,
                    labeled_dir,
                    image_size=224,
                    batch_size=32,
                    seed = 1234,
                    val_split = 0.2):
    """prepares the unlabeled data and the labeled data for ingestion into simclr architecture
    
    Args:
        unlabeled_dir - directory to unlabeled data
        labled_dir - directory to labeled_data
        
    Warnings:
        utilizes image_dataset_from_directory, so unlabeled data may need additional level of abstraction. Can also point to a normal directory with N subdirectories (Where N = number of classes) and set the labels = None option
        
    Returns:
        unlabeled_train_dataset - a tf image dataset of unlabeled images. Created with Labels = None
        labeled_train_dataset
    """
    unlabeled_train_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            unlabeled_dir,
            shuffle = True,
            color_mode = 'rgb',
            image_size = (image_size, image_size),
            batch_size = batch_size))


    labeled_train_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            labeled_dir,
            shuffle = True,
            seed = seed,
            color_mode = 'rgb',
            validation_split = val_split,
            subset = 'training',
            image_size = (image_size, image_size),
            batch_size = batch_size))


    test_dataset = (
        tf.keras.utils.image_dataset_from_directory(
            labeled_dir,
            shuffle = True,
            seed = seed,
            validation_split = val_split,
            subset = 'validation',
            color_mode = 'rgb',
            image_size = (image_size, image_size),
            batch_size = batch_size))
    
    
    train_dataset = tf.data.Dataset.zip((unlabeled_train_dataset, labeled_train_dataset)).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, labeled_train_dataset, test_dataset