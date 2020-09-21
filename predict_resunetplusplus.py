import os

import numpy as np
from keras.optimizers import SGD
from tensorflow.keras.metrics import Precision, Recall
from train_resunetplusplus import ResUnetPlusPlus, preprocess

import constant
from data_creation import load_test_data, load_test_data_miccai
from loss import dice_coef
import tensorflow as tf

from keras import backend as K


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
K.set_image_data_format('channels_last')

img_rows = constant.IMAGE_ROW
img_cols = constant.IMAGE_COL
weight_path = constant.WEIGHT_PATH_RESUNETPLUSPLUS

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
    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    imgs_test, mask = load_test_data()
    imgs_test = preprocess(imgs_test)
    mask = preprocess(mask)

    imgs_test = imgs_test.astype('float32')

    mean = np.mean(imgs_test)
    std = np.std(imgs_test)

    imgs_test -= mean
    imgs_test /= std

    mask = mask.astype('float32')

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)


    arch = ResUnetPlusPlus(input_size=256)
    model = arch.build_model()
    model.summary()
    model.load_weights(weight_path)

    optimizer = SGD(lr=1e-5, momentum=0.9, nesterov=True)
    metrics = [dice_coef, Recall(), Precision()]
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)

    imgs_mask_test_result = model.predict(imgs_test, verbose=1)

    # For image that has been flipped unintentionally
    for k in range(len(imgs_mask_test_result)):
        imgs_mask_test_result[k][:, :, :] = imgs_mask_test_result[k][:, ::-1, :]

    result = get_metrics(mask, imgs_mask_test_result)

    print(result)


if __name__ == '__main__':
    predict()
