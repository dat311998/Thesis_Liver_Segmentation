from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Add, Concatenate, Multiply, Reshape, Dense
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from skimage.transform import resize
from tensorflow.keras.metrics import Precision, Recall

import constant
from data_creation import load_train_data
from loss import dice_coef

K.set_image_data_format('channels_last')

img_rows = constant.IMAGE_ROW
img_cols = constant.IMAGE_COL
experiment_path = constant.EXPERIMENT_PATH
weight_path = constant.WEIGHT_PATH

weight_decay = 1e-2


"""
- Author DebeshJha
- Link https://github.com/DebeshJha/ResUNetplusplus_with-CRF-and-TTA
"""


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides, kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", kernel_regularizer=l2(weight_decay))(x)

    ## Shortcut
    s = Conv2D(n_filter, (1, 1), padding="same", strides=strides, kernel_regularizer=l2(weight_decay))(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides, kernel_regularizer=l2(weight_decay))(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1, kernel_regularizer=l2(weight_decay))(x)

    ## Shortcut
    s = Conv2D(n_filter, (1, 1), padding="same", strides=strides, kernel_regularizer=l2(weight_decay))(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="SAME", kernel_regularizer=l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="SAME", kernel_regularizer=l2(weight_decay))(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="SAME", kernel_regularizer=l2(weight_decay))(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding="SAME", kernel_regularizer=l2(weight_decay))(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="SAME", kernel_regularizer=l2(weight_decay))(y)
    return y


def attention_block(g, x):
    # g: Output of Parallel Encoder block
    # x: Output of Previous Decoder block

    filters = x.shape[-1]

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv2D(filters, (3, 3), padding="SAME", kernel_regularizer=l2(weight_decay))(g_conv)

    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(filters, (3, 3), padding="SAME", kernel_regularizer=l2(weight_decay))(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv2D(filters, (3, 3), padding="SAME", kernel_regularizer=l2(weight_decay))(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul


class ResUnetPlusPlus:
    def __init__(self, input_size=256):
        self.input_size = input_size

    def build_model(self):
        n_filters = [32, 64, 128, 256, 512]
        inputs = Input((self.input_size, self.input_size, 1))

        c0 = inputs
        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)

        ## Bridge
        b1 = aspp_block(c4, n_filters[4])

        ## Decoder
        d1 = attention_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attention_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attention_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])

        ## output
        outputs = aspp_block(d3, n_filters[0])
        outputs = Conv2D(1, (1, 1), padding="same", kernel_regularizer=l2(weight_decay))(outputs)
        outputs = Activation("sigmoid")(outputs)

        ## Model
        model = Model(inputs, outputs)

        return model


def preprocess(imgs):
    print('--- Preprocessing images ---')
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


# Train ResUNet++
def train():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    # TRAINING IMAGES
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

    arch = ResUnetPlusPlus(input_size=256)
    model = arch.build_model()
    # model.summary()
    optimizer = SGD(lr=1e-5, momentum=0.9, nesterov=True)
    metrics = [dice_coef, Recall(), Precision()]

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)

    model_checkpoint = ModelCheckpoint(experiment_path + '/' + 'weights-resunet++.{epoch:02d}-{loss:.2f}.h5', monitor='val_dice_coef',
                                       save_best_only=True, save_weights_only=False, mode='max', period=10)
    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=10,
                        epochs=130, verbose=1,
                        validation_split=0.2, shuffle=True,
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
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    train()