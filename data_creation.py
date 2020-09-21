import os
import numpy as np
import nibabel
import constant
from sklearn.utils import shuffle

training_folder = constant.PROCESSED_TRAINING_DATA
testing_folder = constant.TESTING_DATA
numpy_folder = constant.NUMPY_DATA
image_rows = constant.IMAGE_ROW
image_cols = constant.IMAGE_COL


# Load up LiTS dataset, decrease the sample size down to 256x256, eliminate non-liver slices and save as numpy array file.
def create_train_data():
    print('-' * 30)
    print('Creating training data...')
    print('-' * 30)
    images = os.listdir(training_folder)

    imgs_train = []
    masks_train = []

    training_masks_file = []
    training_images_file = []

    for idx in range(131):
        for item in images:
            if ('segmentation-' + str(idx) + '.nii') in item:
                training_masks_file.append(item)

    for idx in range(131):
        for item in images:
            if ('volume-' + str(idx) + '.nii') in item:
                training_images_file.append(item)

    training_images_file, training_masks_file = shuffle(training_images_file, training_masks_file, random_state=42)

    for liver, orig in zip(training_masks_file, training_images_file):
        print('Processing: ' + liver)
        print('Processing: ' + orig)

        training_image = nibabel.load(os.path.join(training_folder, orig))
        training_mask = nibabel.load(os.path.join(training_folder, liver))

        for k in range(training_mask.shape[2]):
            image_2d = np.array(training_image.get_fdata()[::2, ::2, k])
            mask_2d = np.array(training_mask.get_fdata()[::2, ::2, k])

            if len(np.unique(mask_2d)) != 1:
                masks_train.append(mask_2d)
                imgs_train.append(image_2d)

    print("Total training slices after eliminate non-liver slice: " + str(len(masks_train)))

    imgs = np.ndarray((len(imgs_train), image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((len(masks_train), image_rows, image_cols), dtype=np.uint8)

    for index, img in enumerate(imgs_train):
        imgs[index, :, :] = img

    for index, img in enumerate(masks_train):
        imgs_mask[index, :, :] = img

    np.save(numpy_folder + '/' + 'imgs_train.npy', imgs)
    np.save(numpy_folder + '/' + 'masks_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    print('--- Loading training images ---')
    imgs_train = np.load(numpy_folder + '/' + 'imgs_train.npy')
    masks_train = np.load(numpy_folder + '/' + 'masks_train.npy')
    return imgs_train, masks_train


if __name__ == '__main__':
    create_train_data()