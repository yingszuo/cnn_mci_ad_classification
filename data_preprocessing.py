# -*- coding: utf-8 -*-
"""
This file contains code of data preprocessing. 
The MRI image data was downloaded in 7 zip folders, after unzipping the folders, 
the preprocessing of images starts with the following code. 

The experiment data (2834 images) is made up of 137 complete 1Yr 1.5T images, 
515 complete 2Yr 1.5T images, and 2181 complete 3Yr 1.5T images. 

"""
#%%
# setting up work environment
import os
import re
import numpy as np
import pandas as pd
import nibabel as nib 
import matplotlib.pyplot as plt
from os import chdir, listdir, remove, walk
from sklearn.model_selection import train_test_split

chdir('C:/Users/Yings/OneDrive/Desktop/Coursework/Master/Thesis_2020/Thesis/code') 
#print(getcwd())

# import all the defined functions
from functions_preprocessing import *

#%%
"""extracting MRI images from invidiauls unzipped folders to store in a folder"""

yr1 = "D:/ADNI_unzipped/raw/Data_yr1"
yr2 = "D:/ADNI_unzipped/raw/Data_yr2"
yr3_p1 = "D:/ADNI_unzipped/raw/Data_yr3_P1"
yr3_p2 = "D:/ADNI_unzipped/raw/Data_yr3_P2"
yr3_p3 = "D:/ADNI_unzipped/raw/Data_yr3_P3"
yr3_p4 = "D:/ADNI_unzipped/raw/Data_yr3_P4"
yr3_p5 = "D:/ADNI_unzipped/raw/Data_yr3_P5"

# define a path to store extracted raw MRI images
rawimages = "D:/ADNI_unzipped/Images"

extract(yr1, rawimages)  
extract(yr2, rawimages)  
extract(yr3_p1, rawimages)  
extract(yr3_p2, rawimages)  
extract(yr3_p3, rawimages)  
extract(yr3_p4, rawimages) 
extract(yr3_p5, rawimages)  


#%%
""" filtering the raw images to 1. remove repeated scans 
collected form 1 person on the same day,
2. remove images that were preprocessed differently by ADNI 
(also are images which have a differnt head orientation)""" 
 
# 1. remove repeated scans
remove_repeat(rawimages)

# 2. remove images preprocessed differently by ADNI (different head orientations)

# define a path to store the filtered images
filtered = "D:/ADNI_unzipped/filtered"

filter_img(rawimages, filtered) 
#print(len(listdir(filtered)))

################################################  Data Preprocessing of original data  #####################################
#%%
"""Proprocessing of original images. 
Go through the images in filtered, and perform 
skull-stripping, brain cropping, resizing and intensity-normalization"""

# define a path to store the processed images
processed = "D:/ADNI_unzipped/processed"

# preprocessing all filtered images
counter = 0
for root, dirs, files in walk(filtered):
    for name in files:
        file_path = root + "/" + name
        img = np.load(file_path)
        processed_img = preprocessing_org(img)
        counter += 1
        new_name = name.replace(".npy", "")
        np.save(processed + '/' + new_name + "_processed", processed_img)
        print(str(counter) + " images have been processed")

#print(len(listdir(processed)))

# check the processed images
img_check = np.load(processed + "/" + listdir(processed)[79])
#print(img_check.shape)

# view the processed brain in 2D & 3D plottings
plt.imshow(img_check[30], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
multi_slice_viewer(img_check, [0, 400, 0, 400])


#%%
"""load and store the subject_id and labels for further data splitting"""
# loading the labels
labels = pd.read_csv('D:/ADNI_unzipped/sub_labels.csv')

# find the unique subject-ids
uq_ids = set(labels['Subject'])

# define a dictionary to store subject_id(keys) and class labels(values)
sub_labels = dict()
for id in uq_ids:
    if id not in sub_labels.keys():
        label = ''.join(np.unique(labels['Group'][labels['Subject'] == id]))
        sub_labels[id] = label
    
# check the ID-label is correct 
#print(sub_labels['027_S_1385'])

#%%
""" split the subjects (subject id) into training, validation and testing"""
subs = [*sub_labels]
labs = [sub_labels[sub] for sub in subs]

# check the distribution of class labels
#print(np.unique(labs, return_counts=True))
    
# split based on subject_id to trainval and test
trainval, test, trainval_y, test_y = train_test_split(subs, labs, stratify = labs, test_size = 0.16, random_state = 87)
# split again to train and validation
train, val, train_y, val_y = train_test_split(trainval, trainval_y, stratify = trainval_y, test_size = 0.20, random_state = 87)

# check the class labels in each dataset
#showpercentage(np.unique(train_y, return_counts=True))   
#print()
#showpercentage(np.unique(val_y, return_counts=True)) 
#print()
#showpercentage(np.unique(test_y, return_counts=True)) 

#%%
"""using the split subjects(subject id) to extract processed images 
into training, validation, and testing datasets """

counter = 0 
# define 6 empty lists to store the image data and their labels
M_train = []
M_test = []
M_val = []

lab_train = []
lab_test = []
lab_val = []

#save the names of the training images, for data gumentation
train_names = []
#go through the procesed images to create train/val/test datasets
for root, dirs, files in walk(processed):
    for name in files:
        file_path = root + "/" + name
        img = np.load(file_path)
        # find the label for the loaded image
        sub_id = "".join(re.findall("ADNI_(\d{3}_S_\d{4})", name))
        lab = sub_labels[sub_id]
        
        # change labels to numeric values
        if lab == "CN":
            new_lab = 0
        if lab == "MCI":
            new_lab = 1
        if lab == "AD":
            new_lab = 2
            
        # store the image and its label        
        if sub_id in train:
            lab_train.append(new_lab)
            M_train.append(img)
            
        elif sub_id in val:
            lab_val.append(new_lab)
            M_val.append(img)
            
        else:
            lab_test.append(new_lab)
            M_test.append(img)
            
        # monitor the process
        counter += 1
        print(str(counter) + " has been added to the input arrays")
        


# define a path to store the input
input_path = "D:/ADNI_unzipped/input/org/"
# save
np.save(input_path + "train_data", np.asarray(M_train))
np.save(input_path + "train_label", np.asarray(lab_train))
np.save(input_path + "val_data", np.asarray(M_val))
np.save(input_path + "val_label", np.asarray(lab_val))
np.save(input_path + "test_data", np.asarray(M_test))
np.save(input_path + "test_label", np.asarray(lab_test))

######################################### Split data randomly######################################################

"""Split all MRI images randomly into training, validation and testing"""
# assigning labels to all processed images and store the labels in a list, lab_all
labels_img = target(processed, sub_labels)

# save the processed images in a list, data
data = []
for root, dirs, files in walk(processed):
    for name in files:
        file_path = root + "/" + name
        img = np.load(file_path)
        data.append(img)

# split all images randomly into training/validation/testing       
trainval_rs, test_rs, trainval_y_rs, test_y_rs = train_test_split(data, labels_img, stratify = labels_img, test_size = 0.16, random_state = 87)
train_rs, val_rs, train_y_rs, val_y_rs = train_test_split(trainval_rs, trainval_y_rs, stratify = trainval_y_rs, test_size = 0.20, random_state = 87)

# define a path to store the randomly split input
input_path2 = "D:/ADNI_unzipped/input/random_split/"

# save
np.save(input_path2 + "train_data", np.asarray(train_rs))
np.save(input_path2 + "train_label", np.asarray(train_y_rs))
np.save(input_path2 + "val_data", np.asarray(val_rs))
np.save(input_path2 + "val_label", np.asarray(val_y_rs))
np.save(input_path2 + "test_data", np.asarray(test_rs))
np.save(input_path2 + "test_label", np.asarray(test_y_rs))



############################################### Data Preprocessing of augmented data  ############################
"""use the defined preprocessing_aug to loop through filtered data to create new training data. 
To create balanced class labels, AD images went through data augmentaion 4 times, 
MCI images went through data augmentaion once, 
and NC images went through data augmentation 2 times"""

filtered = "D:/ADNI_unzipped/filtered"

# define path to store the augmented training data
aug_train = "D:/ADNI_unzipped/aug/"

# set two counter 'pgs',to monitor the progress and
# 'faked', to count how many synthetic images 
pgs = 0
faked= 0

# loop through the training data and perform data augmentation
for root, dirs, files in walk(filtered):
    for name in files:
        # only traiing data willbe processed using data augmentation techniques
        if sub_id in train:
            file_path = root + '/' + name
            img = np.load(file_path)
            new_name = name.replace(".npy", "")
            sub_id = "".join(re.findall("ADNI_(\d{3}_S_\d{4})", name))
            lab = sub_labels[sub_id]
            
            # first preprocessed the filtered train image normally and save it
            processed_img = preprocessing_org(img)
            np.save(aug_train + new_name + "_processed", processed_img)
            
            # progress counter + 1 for each original image loaded
            pgs += 1
            # counter set to 0 for each loaded image 
            counter = 0
            
            # create the synethetic training data
            """ the total synthetic training data should be within 3317, 
            this is computed as 5000 (wanted total taining) - 1683 (normal processed training sampple))"""   
            if faked < 3317:
                if lab == "AD":
                    while counter < 4:
                        affine, col, img_aug = preprocessing_aug(img)  
                        np.save(aug_train + new_name + "_" + affine + "-" + col + "_" + "aug "+ str(counter), img_aug)  
                        counter += 1
                    faked += counter 

                if lab == "CN":
                    while counter < 2:
                        affine, col, img_aug = preprocessing_aug(img)  
                        np.save(aug_train + new_name + "_" + affine + "-" + col + "_" + "aug " + str(counter), img_aug)  
                        counter += 1
                    faked += counter 
        
                if lab == "MCI":
                    while counter < 1:
                        affine, col, img_aug = preprocessing_aug(img)  
                        np.save(aug_train + new_name + "_" + affine + "-" + col + "_" + "aug " + str(counter), img_aug)  
                        counter += 1
                    faked += counter 
                
                # monitor how many synthetic data is created
                print()
                print(str(faked) + " artificial images created")
                
            # monitor how many training samples have been through the augmentation process  
            print()
            print(str(pgs) + " training images has been through the augmentation process")
            

#print(len(aug_train))  
# pick a random image to check the quality
#check(aug_train)
# check the distribution of labels
#count_labels(aug_train, sub_labels)

    
#%%

"""save training data & save copies of the validation & test data"""

# create empty lists for storing validation and test data
M_train_aug = []
lab_train_aug = []

# set counter to 0
counter = 0        
        
for root, dirs, files in walk(aug_train):
    for name in files:
        file_path = root + name
        img = np.load(file_path)
        sub_id = "".join(re.findall("ADNI_(\d{3}_S_\d{4})", name))
        lab = sub_labels[sub_id]
        
        if lab == "CN":
            new_lab = 0
        if lab == "MCI":
            new_lab = 1
        if lab == "AD":
            new_lab = 2            
        
        # save training data
        M_train_aug.append(img)
        lab_train_aug.append(new_lab)
        
        # monitor the process       
        counter += 1
        print(str(counter) + " has been added to the input array")  


# define a path to store the augmented training input
input_path3 = "D:/ADNI_unzipped/input/aug/"
np.save(input_path3 + "train_data", np.asarray(M_train_aug))
np.save(input_path3 + "train_label", np.asarray(lab_train_aug))
# save copies of the validation and test data that was processed normally 
np.save(input_path3 + "val_data", np.asarray(M_val))
np.save(input_path3 + "val_label", np.asarray(lab_val))
np.save(input_path3 + "test_data", np.asarray(M_test))
np.save(input_path3 + "test_label", np.asarray(lab_test))








###############################################  Supplementary code ###########################################
# selected a random image to check the outcome of each preprocessing_org steps
icheck = np.load(filtered + "/" + listdir(filtered)[16])
# check skull-stripping
iskull = skull_stripper(icheck)
# view in 2D plot 
plt.imshow(iskull[128], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
# multislice viewer
multi_slice_viewer(iskull, [0, 400, 0, 400])
# save image to view in 3D space
img_skull = nib.Nifti1Image(iskull, affine=np.eye(4))
img_skull.header.get_xyzt_units()
img_skull.to_filename(os.path.join('D:/ADNI_unzipped/view','img_skull.nii')) 

#check brain cropping
icrop = crop(iskull)
plt.imshow(icrop[80], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
# multislice viewer
multi_slice_viewer(icrop, [0, 400, 0, 400])
# save image to view in 3D space
img_crop = nib.Nifti1Image(icrop, affine=np.eye(4))
img_crop.header.get_xyzt_units()
img_crop.to_filename(os.path.join('D:/ADNI_unzipped/view','img_crop.nii')) 

# check resizing 
istd = resizer(icrop)
plt.imshow(icrop[30], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
# multislice viewer
multi_slice_viewer(istd, [0, 400, 0, 400])
# save image to view in 3D space
img_std = nib.Nifti1Image(istd, affine=np.eye(4))
img_std.header.get_xyzt_units()
img_std.to_filename(os.path.join('D:/ADNI_unzipped/view','img_std.nii')) 


# checking intensity-normalization/the final products
inor = scaler(istd)
plt.imshow(inor[30], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
# multislice viewer
multi_slice_viewer(inor, [0, 400, 0, 400])
# save image to view in 3D space
img_nor = nib.Nifti1Image(inor, affine=np.eye(4))
img_nor.header.get_xyzt_units()
img_nor.to_filename(os.path.join('D:/ADNI_unzipped/view','img_nor.nii')) 



#%% remove the images failed to pass the preprocessing_org
# loaded the images that failed with the preprocessing (error message) - image 80
f_img = np.load(processed + "/" + listdir(processed)[80])
# double check the quality of the failed image in 2D and 3D view
plt.imshow(f_img[128], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
multi_slice_viewer(f_img, [0, 600, 0, 600])
# remove the the image
remove(processed + "/" + listdir(processed)[80]) 


#%%
# Check the quality of each data augmentation technique

# affine atransformation is processed after skull-stripping before brain cropping
img = np.load(filtered + '/' + listdir(filtered)[84])
img = skull_stripper(img)
print(img.shape)
#plt.imshow(img[120], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
multi_slice_viewer(img, [0, 600, 0, 600]) 

"""rotation"""
# rotate a slice of a MRI image
from scipy.ndimage.interpolation import rotate
plt.imshow(img[120,:,:], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
img_rot2d = rotate(img, -15, mode='nearest', axes=(0, 1), reshape=False)
plt.imshow(img[120], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")

# check the rotated 3D image
img_rot = random_rotation_3d(img)
plt.imshow(img_rot[120], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
plt.imshow(img_rot[:,110,:], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
plt.imshow(img_rot[:,:,80], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")

multi_slice_viewer(img, [0, 600, 0, 600]) 
multi_slice_viewer(img_rot, [0, 600, 0, 600]) 

"""flip"""
# check flipped 3D image
img_flip = flip(img)
plt.imshow(img[120], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
plt.imshow(img_flip[121], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")

multi_slice_viewer(img, [0, 600, 0, 600]) 
multi_slice_viewer(img_flip, [0, 600, 0, 600]) 

"""elastic transformation"""
# check a slice of MRI images after elastic transformation
plt.imshow(img[120], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
slice_ela = elastic_transform(img[120])
plt.imshow(slice_ela, aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")

# check a 3D image after elastic transformation
img_ela = elastic_3d(img)
plt.imshow(img_ela[120], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
plt.imshow(img_ela[:,110,:], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
plt.imshow(img_ela[:,:,80], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")

multi_slice_viewer(img, [0, 600, 0, 600]) 
multi_slice_viewer(img_ela, [0, 600, 0, 600]) 


"""contrast and sharpness"""
# pixel-level transformation are processed after intensity normalization
preprocessed  = "D:/ADNI_unzipped/processed"
img2 = np.load(preprocessed + '/' + listdir(preprocessed)[571])
print(img2.shape)
#plt.imshow(img2[32], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
multi_slice_viewer(img2, [0, 600, 0, 600]) 

# check a slice of MRI image with changing contrast and sharpness
img_sharp = sharp(img2[32], 1.98)
img_contra = contra(img2[32], 1.89)
plt.imshow(img2[32], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
plt.imshow(img_sharp, aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
plt.imshow(img_contra, aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")

# check a 3D image after chaing contrast and sharpness
img_con3d = contra_3D(img2)
img_sa3d = sharp_3D(img2)
plt.imshow(img_con3d[:,:,28], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
plt.imshow(img_sa3d[:,30,:], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")

multi_slice_viewer(img2, [0, 600, 0, 600]) 
multi_slice_viewer(img_con3d, [0, 600, 0, 600]) 
multi_slice_viewer(img_sa3d, [0, 600, 0, 600]) 


"""noise"""
# checking adding noise to a 2D slice
from skimage.util import random_noise
img_noisy = random_noise(img2[32], mode='speckle', seed=None)
plt.imshow(img_noisy, aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")

# check a #D image after adding in noise
img_no = noise(img2)
plt.imshow(img_no[:,25,:], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")
plt.imshow(img_no[:,:,27], aspect=0.5, extent = [0, 200, 0, 400], cmap = "gray")

multi_slice_viewer(img2, [0, 600, 0, 600]) 
multi_slice_viewer(img_no, [0, 600, 0, 600]) 














