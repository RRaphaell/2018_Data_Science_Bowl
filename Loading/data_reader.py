import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm


def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def read_mask(directory, target_size=None):
    """Read and resize masks contained in a given directory."""
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask_tmp = read_image(mask_path, cv2.IMREAD_GRAYSCALE, target_size)
        if not i:
            mask = mask_tmp
        else:
            mask = np.maximum(mask, mask_tmp)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def read_train_data_properties(train_dir, img_dir_name, mask_dir_name):
    """Read basic properties of training images and masks"""

    df_list = []
    images = []

    for dir_name in tqdm(next(os.walk(train_dir))[1]):
        img_dir = os.path.join(train_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img = read_image(img_path)
        img_shape = img.shape
        mask_dir = os.path.join(train_dir, dir_name, mask_dir_name)
        num_masks = len(next(os.walk(mask_dir))[2])
        images.append(img)
        df_list.append([img_name_id, img_shape[0], img_shape[1], num_masks, img_path, mask_dir])

    train_df = pd.DataFrame(df_list, columns=['id', 'height', 'width', 'num_masks', 'image_path', 'mask_dir'])
    return train_df, images


def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""

    df_list = []
    images = []

    for dir_name in tqdm(next(os.walk(test_dir))[1]):
        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img = read_image(img_path)
        img_shape = img.shape
        images.append(img)
        df_list.append([img_name_id, img_shape[0], img_shape[1], img_path])

    test_df = pd.DataFrame(df_list, columns=['id', 'height', 'width', 'image_path'])
    return test_df, images
