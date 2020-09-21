# Thesis_Liver_Segmentation
Repository for Thesis: Liver Segmentation using Fully Convolutional Neural Networks

In this repository, I perform training and inference for liver segmentation using 2D ResUNet model in CT scans.
- Link for LiTS dataset: https://drive.google.com/drive/folders/0B0vscETPGI1-Q1h1WFdEM2FHSUE.
- Link for 3DIRCAD-01 dataset: https://www.dropbox.com/s/8h2avwtk8cfzl49/ircad-dataset.zip?dl=0.
- Link for trained weight of ResUNet and ResUNet++: https://drive.google.com/drive/folders/1RTlm9pYuil787zOoIqMeLbPSJBCnpMIg?usp=sharing

Before training and prediction:
- Training needs GPU, CPU won't be able to sufficient. My training stage was done on GX2080ti with 11GB and 62GB Memory Disk. Therefore, the configuration is defined as to maximize the use of GPU, change it upon your available hardware.
- If you use IDE such as Pycharm, go to Project Structure and set main folder is source folder in order to import custom modules.
- Folders are ordered with intention, please modify these so that it fits your folder paths.
- Training for ResUNet takes about 18 hours (100 epochs) using above configurations. Training for ResUNet++ takes about 3 days (130 epochs)

In order to train and predict, you need to:
- Step 1: Download the datasets and put on appropriate folders.
- Step 2: Run data_preprocessing.py and note the comments I made in the code in order to run the code without errors.
- Step 3: Run data_creation.py. This will create train numpy files (3D images and masks), 3DIRCAD test data (3D images and masks), MICCAI test data with specific range (3D images and masks).
- Step 4: Run train.py for ResUNet and train_resunetplusplus.py for ResUNet++
- Step 5: In order to perform inference stage, you can use the available weight file (weights.100-0.01.h5 for ResUNet and weights-resunet++.h5 for ResUNet++), then run predict.py for ResUNet or predict_resunetplusplus.py for ResUNet++. There is snippet of code where I use to flip the images because predict function in Keras somehow randomly flipped the result vertically, such cases are CT scans number 2, 4, 6, 7, 8, 10, 11, 18, 19. For others, please comment out these lines for better prediction result. 
