from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Add, Concatenate
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from loss import dice_coef
from tensorflow.keras.metrics import Precision, Recall
from skimage.transform import resize

import constant
from data_creation import load_train_data

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = constant.IMAGE_ROW
img_cols = constant.IMAGE_COL
experiment_path = constant.EXPERIMENT_PATH
weight_path = constant.WEIGHT_PATH


def bn_act(x, act=True):
    x = BatchNormalization()(x)
    if act == True:
        x = Activation("relu")(x)
    return x


def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = Add()([conv, shortcut])
    return output


def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = Add()([shortcut, res])
    return output


def upsample_concat_block(x, xskip):
    u = UpSampling2D((2, 2))(x)
    c = Concatenate()([u, xskip])
    return c


def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = Input((img_rows, img_cols, 1))

    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)

    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)

    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)

    model = Model(inputs=[inputs], outputs=[outputs])

    metrics = [dice_coef, Recall(), Precision()]

    model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=metrics)

    return model


# Add new dimension to the data so that it can be fed into the network
def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    # Loading and preprocessing training data
    imgs_train, imgs_mask_train = load_train_data()
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model = ResUNet()

    model_checkpoint = ModelCheckpoint(experiment_path + '/' + 'weights.{epoch:02d}-{loss:.2f}.h5', monitor='val_loss',
                                       save_best_only=False, save_weights_only=False, mode='min', period=50)
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=20,
                        epochs=200, verbose=1,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[model_checkpoint])

    # Saving our predictions in the directory 'preds'
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model dice coeff')
    plt.ylabel('Dice coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    train()