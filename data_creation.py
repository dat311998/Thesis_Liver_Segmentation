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


# Create test data, do the same steps as the train dataset creation.
# In order to test out the model fully automatic, disable the non-liver slice filter function.
def create_test_data():
    print('-' * 30)
    print('Creating test data...')
    print('-' * 30)

    images = os.listdir(testing_folder)
    images = np.sort(images)

    imgs_test = []
    masks_test = []

    testing_images_files = []
    testing_masks_files = []

    temp_img = []
    temp_mask = []

    for idx in range(1):
        for item in images:
            if 'orig' in item:
                testing_images_files.append(item)

    for idx in range(1):
        for item in images:
            if 'liver' in item:
                testing_masks_files.append(item)

    # Test the CT scan individually
    temp_img.append(testing_images_files[4])
    temp_mask.append(testing_masks_files[4])

    for orig, liver in zip(temp_img, temp_mask):
        print('Processing: ' + orig)
        print('Processing: ' + liver)
        testing_image = nibabel.load(os.path.join(testing_folder, orig))
        testing_mask = nibabel.load(os.path.join(testing_folder, liver))
        print("Total testing slices before eliminate non-liver slice: " + str(testing_image.shape[2]))
        for k in range(testing_image.shape[2]):
            image_2d = np.array(testing_image.get_fdata()[::2, ::2, k])
            mask_2d = np.array(testing_mask.get_fdata()[::2, ::2, k])
            if len(np.unique(mask_2d)) != 1:
                masks_test.append(mask_2d)
                imgs_test.append(image_2d)

    print("Total testing slices after eliminate non-liver slice: " + str(len(imgs_test)))

    imgst = np.ndarray((len(imgs_test), image_rows, image_cols), dtype=np.uint8)
    maskst = np.ndarray((len(masks_test), image_rows, image_cols), dtype=np.uint8)

    for index, img in enumerate(imgs_test):
        imgst[index, :, :] = img

    for index, mask in enumerate(masks_test):
        maskst[index, :, :] = mask

    np.save(numpy_folder + '/' + 'imgs_test.npy', imgst)
    np.save(numpy_folder + '/' + 'imgs_mask.npy', maskst)
    print('Saving to .npy files done.')


def load_test_data():
    print('--- Loading test images ---')
    imgs_test = np.load(numpy_folder + '/' + 'imgs_test.npy')
    masks_test = np.load(numpy_folder + '/' + 'imgs_mask.npy')
    return imgs_test, masks_test


# Create test data for MICCAI dataset
def create_test_data_micaai():
    print('-' * 30)
    print('Creating test data...')
    print('-' * 30)

    imgs_test = []
    masks_test = []

    testing_images_files = []
    testing_masks_files = []

    # Modify the test folder before generate test
    item_list = os.listdir(testing_folder)
    np.sort(item_list)

    # Modify CT scan range when needed
    for idx in range(10):
        testing_images_files.append('volume-' + str(idx) + '.nii')
        testing_masks_files.append('segmentation-' + str(idx) + '.nii')

    for orig, liver in zip(testing_images_files, testing_masks_files):
        print('Processing: ' + orig)
        print('Processing: ' + liver)
        testing_image = nibabel.load(os.path.join(testing_folder, orig))
        testing_mask = nibabel.load(os.path.join(testing_folder, liver))
        print("Total testing slices before eliminate non-liver slice: " + str(testing_image.shape[2]))
        for k in range(testing_image.shape[2]):
            image_2d = np.array(testing_image.get_fdata()[::2, ::2, k])
            mask_2d = np.array(testing_mask.get_fdata()[::2, ::2, k])
            # Disable if testing with uncropped data
            if len(np.unique(mask_2d)) != 1:
                masks_test.append(mask_2d)
                imgs_test.append(image_2d)

    print("Total testing slices after eliminate non-liver slice: " + str(len(imgs_test)))

    imgst = np.ndarray((len(imgs_test), image_rows, image_cols), dtype=np.uint8)
    maskst = np.ndarray((len(masks_test), image_rows, image_cols), dtype=np.uint8)

    for index, img in enumerate(imgs_test):
        imgst[index, :, :] = img

    for index, mask in enumerate(masks_test):
        maskst[index, :, :] = mask

    np.save(numpy_folder + '/' + 'imgs_test_miccai.npy', imgst)
    np.save(numpy_folder + '/' + 'imgs_mask_miccai.npy', maskst)
    print('Saving to .npy files done.')


# Load test data for MICCAI dataset
def load_test_data_miccai():
    print('--- Loading miccai test images ---')
    imgs_test = np.load(numpy_folder + '/' + 'imgs_test_miccai.npy')
    masks_test = np.load(numpy_folder + '/' + 'imgs_mask_miccai.npy')
    return imgs_test, masks_test


if __name__ == '__main__':
    create_train_data()
    create_test_data()
    create_test_data_micaai()