from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from keras import backend as K

import constant
from data_creation import load_test_data
from train import ResUNet, preprocess

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
K.set_image_data_format('channels_last')

img_rows = constant.IMAGE_ROW
img_cols = constant.IMAGE_COL
weight_path = constant.WEIGHT_PATH

smooth = 1.


def get_dice_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def get_recall(y_true, y_pred):
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    m = tf.keras.metrics.Recall()
    m.update_state(y_true, y_pred)
    r = m.result().numpy()
    m.reset_states()
    return r


def get_precision(y_true, y_pred):
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    m = tf.keras.metrics.Precision()
    m.update_state(y_true, y_pred)
    r = m.result().numpy()
    m.reset_states()
    return r


def get_metrics(y_true, y_pred):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    dice_coef_val = get_dice_coef(y_true, y_pred)

    y_true = y_true.astype(np.int32)

    recall_value = get_recall(y_true, y_pred)
    precision_value = get_precision(y_true, y_pred)

    return [dice_coef_val, recall_value, precision_value]


def predict():
    for idx in range(1):
        print('-' * 30)
        print('Loading model and preprocessing test data...' + str(idx))
        print('-' * 30)

        model = ResUNet()
        model.load_weights(weight_path)

        #  Load and preprocessing test dataset
        img_test, mask = load_test_data()

        img_test = preprocess(img_test)
        img_test = img_test.astype('float32')

        mean = np.mean(img_test)  # mean for data centering
        std = np.std(img_test)  # std for data normalization

        img_test -= mean
        img_test /= std

        mask = preprocess(mask)
        mask = mask.astype('float32')

        print('-' * 30)
        print('Predicting masks on test data...' + str(idx))
        print('-' * 30)

        imgs_mask_test_result = model.predict(img_test, verbose=1)

        # For image that has been flipped unintentionally
        for k in range(len(imgs_mask_test_result)):
            imgs_mask_test_result[k][:, :, :] = imgs_mask_test_result[k][:, ::-1, :]

        result = get_metrics(mask, imgs_mask_test_result)

        print(result)