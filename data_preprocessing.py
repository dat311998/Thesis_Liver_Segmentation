import copy

import numpy as np
from medpy.io import load, save
from scipy.ndimage import rotate

import constant

training_folder = constant.TRAINING_DATA
testing_folder = constant.TESTING_DATA
processed_training_folder = constant.PROCESSED_TRAINING_DATA
processed_testing_folder = constant.PROCESSED_TESTING_DATA


def flip_image(imgs, masks):
    reflected_img = copy.deepcopy(imgs)
    reflected_img = np.fliplr(reflected_img)

    reflected_mask = copy.deepcopy(masks)
    reflected_mask = np.fliplr(reflected_mask)

    return reflected_img, reflected_mask


def rotate_img(imgs, masks):
    rotated_img = copy.deepcopy(imgs)
    rotated_img = rotate(rotated_img, 10, reshape=False)

    rotated_mask = copy.deepcopy(masks)
    rotated_mask = rotate(rotated_mask, 10, reshape=False)

    return rotated_img, rotated_mask


# Clipping Hounsfield value for better liver recognition and less noise
def truncate_HU_value(range1, range2, img_path, save_path):
    print("*** Truncating HU value to eliminate superfluous information on training data ***")
    for idx in range(range1, range2):
        img, img_header = load(img_path + '/' + 'volume-' + str(idx) + '.nii')
        img[img < -200] = -200
        img[img > 250] = 250
        img = np.array(img, dtype='int16')
        print('Saving image ' + str(idx))
        save(img, save_path + '/' + 'volume-' + str(idx) + '.nii')


# Remove the tumor label if the dataset is with 3 labels
def remove_tumor_label(range1, range2, img_path, save_path):
    print("*** Removing tumor label ***")
    for idx in range(range1, range2):
        img, img_header = load(img_path + '/' + 'segmentation-' + str(idx) + '.nii')
        img[img == 2] = 1
        img = np.array(img, dtype='uint8')
        print('Saving image ' + str(idx))
        save(img, save_path + '/' + 'segmentation-' + str(idx) + '.nii')


def create_augmented_data(range1, range2, img_path, save_path):
    print("*** Creating augmented data ***")
    for idx in range(range1, range2):
        img, img_header = load(img_path + '/' + 'volume-' + str(idx) + '.nii')
        mask, mask_header = load(img_path + '/' + 'segmentation-' + str(idx) + '.nii')
        img, mask = flip_image(imgs=img, masks=mask)
        img, mask = rotate_img(imgs=img, masks=mask)
        img = np.array(img, dtype='int16')
        mask = np.array(mask, dtype='uint8')
        print('Saving image ' + str(idx))
        save(img, save_path + '/' + 'aug_volume-' + str(idx) + '.nii')
        print('Saving mask ' + str(idx))
        save(mask, save_path + '/' + 'aug_segmentation-' + str(idx) + '.nii')


truncate_HU_value(range1=71, range2=131, img_path=training_folder, save_path=processed_training_folder)
remove_tumor_label(range1=71, range2=131, img_path=training_folder, save_path=processed_training_folder)
create_augmented_data(range1=0, range2=131, img_path=processed_training_folder, save_path=processed_training_folder)