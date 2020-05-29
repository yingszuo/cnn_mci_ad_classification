# -*- coding: utf-8 -*-
"""
functions fr image preprocessing
"""

#%%
# import packages 
import re
import random
import shutil
import numpy as np
import nibabel as nib 
from random import uniform
from deepbrain import Extractor
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.transform import resize
from skimage.util import random_noise
from skimage.measure import label, regionprops
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.ndimage.filters import gaussian_filter
from os import listdir, walk, remove    


#%%
# define a function to go through folders to extract out all nii files (MRI images). 
def extract(origin, destination):
    counter = 0 
    for root, dirs, files in walk(origin):
        for name in files:
            if name.endswith((".nii")):
                file_path = root + "/" + name
                counter += 1
                shutil.move(file_path, destination)
    return(str(counter) + " files have been moved to selected folder")
    
#%%
# define a function to filter out the repeatative scans collected from the same person on the same day
def remove_repeat(path):
    counter = 0 
    sub_year = set()
    for root, dirs, files in walk(path):
        for name in files:
            file_path = root + "/" + name
            # find the subject_id of the image
            sub_id = "".join(re.findall("ADNI_(\d{3}_S_\d{4})", name))
            # find the date of collecting the image 
            date = int(re.findall("20\d{6}", name)[0])
            # create a unique subject + date variable
            id_year = sub_id + "_" +str(date)    
            # remove the repeated scan collected from one person on one day  
            if id_year in sub_year:
                counter += 1
                # remove the repeated images
                remove(file_path) 
                print(str(counter) + " repeated images have been removed")
            else: 
                sub_year.add(id_year)
    


#%%
"""images preprocessed differently by ADNI also have a different head orientation, 
so they were removed. This is done by filtering the dimensions. 
By intensive checking, it is found that if the last dimension s[2] 
is larger than the first dimension s[0], then the head orientation in this image 
is different to the majority, also the image was preprocessed differntly 
from ADNI(can be seen from the name ofthe file). These images will not saved to the filtered image folder"""     
           
def filter_img(path, des):
    files = listdir(path)
    counter = 0 
    for name in files:
        fpath = path + "/" + name
        nii_img = nib.load(fpath)
        try:
            data = nii_img.get_data()
            img = np.asarray(data)
            s = img.shape  
            if s[0] > s[2]:
                name = name.replace(".nii", "") 
                newpath = des + "/" + name
                np.save(newpath,data)
            else: 
                counter += 1
        except: 
            counter += 1
    print(str(counter) + " images not meeting the selection criteria/failed to load")
  
#%%
# define functions for preprocessing of original data
    
# skukk-stripping
def skull_stripper(image):
    ext = Extractor()
    prob = ext.run(image)
    # set the threshold for extractor to extract the brain
    mask = prob < 0.7
    img_filtered = np.ma.masked_array(image, mask = mask)
    # fill the background to 0 
    img_filtered = img_filtered.filled(0)

    return img_filtered

# brain-cropping    
def crop(image):
  img_bo = image > 0
  img_labeled = label(img_bo)
  bounding_box = regionprops(img_labeled)
  # get the minimum and maximum valyues of x, y, z axis
  bb= bounding_box[0].bbox
  #print(len(bounding_box))
  #print(bb[0],bb[3],bb[1],bb[4],bb[2],bb[5])
  
  #crop the image along the x, yz and z axis
  img_crop= image[bb[0]:bb[3],
                  bb[1]:bb[4],
                  bb[2]:bb[5]]
  #print(img_crop.shape)
  return img_crop

# resize the images to (64,64,64)
def resizer(image, ideal_shape = (64, 64, 64)):
    # go along the x axis,resize images on the y and z axis
    img_new1 = np.zeros((image.shape[0],ideal_shape[1], ideal_shape[2]))
    for i in range(image.shape[0]):
        img = image[i,:,:]
        img_new1[i,:,:] = resize(img, (ideal_shape[1], ideal_shape[2]), anti_aliasing=True)
    
    # go along the y axis, resize images on the x and z axis
    img_new2 = np.zeros(ideal_shape)
    for i in range(img_new1.shape[1]):
        img = img_new1[:,i,:]
        img_new2[:,i,:] = resize(img, (ideal_shape[0], ideal_shape[2]), anti_aliasing=True)
    return img_new2

# intensity-nomalization
def scaler(image): 
  img_f = image.flatten()
  # find the range of the pixel values
  i_range = img_f[np.argmax(img_f)] - img_f[np.argmin(img_f)]
  # clear the minus pixel values in images
  image = image - img_f[np.argmin(img_f)]
  img_normalized = np.float32(image/i_range)
  #print(M_normalized.shape)
  return img_normalized 

# use the individual functions created to create a preprocessing procedure
def preprocessing_org(img):
    img_skull = skull_stripper(img)
    img_crop = crop(img_skull)
    img_std = resizer(img_crop)
    img_nor = scaler(img_std)
    return img_nor

#%%
# defining a function to 3D visualize the MRI images in axonal plane
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
                
def multi_slice_viewer(volume, extent):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index], extent = extent, cmap="gray")
    fig.canvas.mpl_connect('key_press_event', process_key)

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()
    
def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])   
    
#%%
# define a function to assign labels to the images in a folder
def target(path, ref):
    files = listdir(path)
    target = []
    for names in files: 
        sub_id = "".join(re.findall("ADNI_(\d{3}_S_\d{4})", names))
        lab = ref[sub_id]
        if lab == "CN":
            new_lab = 0
        if lab == "MCI":
            new_lab = 1
        if lab == "AD":
            new_lab = 2
        target.append(new_lab)
    print(str(len(target)) + " labels have been assigned to the data")
    return target


#%%
def showpercentage(alist):
    array = np.unique(alist, return_counts=True)
    pad = array[1][0]/np.sum(array[1])
    pcn = array[1][1]/np.sum(array[1])
    pmci = array[1][2]/np.sum(array[1])
    print(str(pad) + " percent of the data has AD label")
    print(str(pcn) + " percent of the data has CN label")
    print(str(pmci) + " percent of the data has MCI label")

#%%
def showp2(alist):
    array = np.unique(alist, return_counts=True)
    pcn = array[1][0]/np.sum(array[1])
    pmci = array[1][1]/np.sum(array[1])
    pad = array[1][2]/np.sum(array[1])
    print(str(pad) + " percent of the data has AD label")
    print(str(pcn) + " percent of the data has CN label")
    print(str(pmci) + " percent of the data has MCI label")   
    

#%%
# define functions for preprocessing of augmented data
    
# 3d image rotation
def random_rotation_3d(img):
    """ Randomly rotate an image by a random angle (-15, 15).
    
    Returns:
    a rotated 3D image
    """
    max_angle = 15
    
    if bool(random.getrandbits(1)):
    # rotate along z-axis
        angle = uniform(-max_angle, max_angle)
        img1 = rotate(img, angle, mode='nearest', axes=(0, 1), order=1, reshape=False)
        #print(angle)
        #print("Z-axis")
    else: 
        img1 = img

    # rotate along y-axis
    if bool(random.getrandbits(1)):
        angle = uniform(-max_angle, max_angle)
        img2 = rotate(img1, angle, mode='nearest', axes=(0, 2), order=1, reshape=False)
        #print(angle)
        #print("Y-axis")
    else:
        img2 = img1
    
    # rotate along x-axis
    if bool(random.getrandbits(1)):
        angle = uniform(-max_angle, max_angle)
        img3 = rotate(img2, angle, mode='nearest', axes=(1, 2), order=1, reshape=False)
        img3 = np.float32(img3)
        #print(angle)
        #print("X-axis")
    else:
        img3 = np.float32(img2)
        
    return img3


# 3d image flipping   
def flip(img):
    axis = random.sample(range(0, 2), 1)[0]
    new_img = np.zeros(img.shape)
    #print(axis)
    if axis == 0:
        for i in range(img.shape[0]):
            new_img[i] = np.fliplr(img[i])
    if axis == 1:
        for i in range(img.shape[1]):
            new_img[:,i,:] = np.fliplr(img[:,i,:])
    # no flip on the Z-axis, as the brain will turn upside down, which is not possible for a normal brain
    new_img = np.float32(new_img)
    
    return new_img   

# elastic transofmraiton on 2d slice
def elastic_transform(image, alpha = 80, sigma = 10, random_state = None, sd = 29):
    """
    alpha = scaling factor the deformation; 
    sigma = smooting factor 
    
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    shape = image.shape
    random_state.seed(sd) 
    #print(random_state.rand(*shape))

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape) 

# elastic transoformation on 3d images
def elastic_3d(img):
    """"  
    using elastic_transform  to transform all slices of one dimension
    seed is fixed per dimension
    transformation happens at one of the three dimensions
    """
    #import random
    axis = random.sample(range(0, 3), 1)[0]
    new_img = np.zeros(img.shape)
    #print(axis)
    if axis == 0:
        # generate a seed for one type of elastic transformation for all images along this axis
        rand = random.sample(range(0, 100), 1)[0]
        for i in range(img.shape[0]):
            new_img[i] = elastic_transform(img[i], sd = rand)
    if axis == 1:
        rand = random.sample(range(0, 100), 1)[0]
        for i in range(img.shape[1]):
            new_img[:,i,:] = elastic_transform(img[:,i,:], sd = rand)
    if axis == 2:
        rand = random.sample(range(0, 100), 1)[0]
        for i in range(img.shape[2]):
            new_img[:,:,i] = elastic_transform(img[:,:,i], sd = rand)
    #print(rand)
    new_img = np.float32(new_img) 
    
    return new_img

# chaing image cotrast for 2d slice
def contra(img, factor):
        # change the pixel values back to 0 - 255
    img = img * 255
    img = Image.fromarray(img)
    img = img.convert('L')
    
    # change contrast
    img1 = ImageEnhance.Contrast(img)
    img_con = img1.enhance(factor)
    
    # converting back to np array
    img_final = np.float32(np.asarray(img_con)/255)
    return img_final
    

# changing image contrast for 3d images    
def contra_3d(img):
    new_img = np.zeros(img.shape)
    """creating contrast factor 
    factors ranging between 0.7 - 1.5 will make the augmented image close to original image"""
    factor = 1      
    while factor < 1.4 and factor > 0.7:
        factor = np.random.uniform(0.4, 1.9)
    #print(factor)   
    # go through each slice to change the contrast and sharpness    
    for i in range(img.shape[0]):
            new_img[i] = contra(img[i], factor)  
            
    return np.float32(new_img)


# changing image sharpness for 2d slice
def sharp(img, factor):
        # change the pixel values back to 0 - 255
    img = img * 255
    img = Image.fromarray(img)
    img = img.convert('L')  
    
     # change sharpness
    img1 = ImageEnhance.Sharpness(img) 
    img_sharp = img1.enhance(factor)
    
    img_final = np.float32(np.asarray(img_sharp)/255)
    return img_final

# changing image sharpness for 3d images
def sharp_3d(img):
    new_img = np.zeros(img.shape)
    """creating contrast factor 
    factors ranging between 0.7 - 1.5 will make the augmented image close to original image"""
    factor = 1      
    while factor < 1.5 and factor > 0.7:
        factor = np.random.uniform(0.4, 2)
    #print(factor)  
    
    for i in range(img.shape[0]):
            new_img[i] = sharp(img[i], factor)  
            
    return np.float32(new_img)   

# adding in noise to 3d images
def noise(img):
    new_img = np.zeros(img.shape)
    # choose one type of noise to add to the image
    mode = ['gaussian', 's&p', 'poisson','speckle']
    random.shuffle(mode)
    idx = random.sample(range(0, 4), 1)[0]
    noise= mode[idx]
    #print(noise)

    # fix a seed for all slices to be added with the exact same noise 
    sd = random.sample(range(0, 100), 1)[0]
    
    new_img = random_noise(img, mode= noise, seed = sd)
        
    return np.float32(new_img)  

# do nothing to the image
def none(img):
    return img

# use the individual functions to create a preprocessing procedure for augmented data
def preprocessing_aug(img):
    affine_functions = [random_rotation_3d, flip, elastic_3d, none]
    col_functions = [contra_3d, sharp_3d, noise, none]
    
    # set both transformations to none
    affine_fun = none
    col_fun = none
    # the training data can needs to pass through a data augmentation technique,
    # it is otherwise a repeated copy of the original preprocessed image.  
    while affine_fun == col_fun:
        # randomly choose an affine transformation
        idx1 = random.sample(range(0, 3), 1)[0]
        random.shuffle(affine_functions)
        affine_fun = affine_functions[idx1]
        # randomly choose a piexel transformation
        idx2 = random.sample(range(0, 3), 1)[0]
        random.shuffle(col_functions)
        col_fun = col_functions[idx2]
    
    # processed images with the new order of steps    
    new_img = scaler(col_fun(scaler(resizer(crop(affine_fun(img))))))
    
    # return the name of the chosen affine transformaiton & pixel-level transformation
    # and the augmented image
    return affine_fun.__name__, col_fun.__name__, new_img    

#%%
# define a function to compute the distribution of class labels in a given folder of images
def count_labels(path, ref_all):
    counter_cn = 0 
    counter_ad = 0 
    counter_mci = 0 
    
    for root, dirs, files in walk(path):
        for name in files:
            if name.endswith((".npy")):
                sub_id = "".join(re.findall("ADNI_(\d{3}_S_\d{4})", name))
                lab = ref_all[sub_id]
                if lab == "CN":
                    counter_cn += 1
                if lab == "MCI":
                    counter_mci += 1
                if lab == "AD":
                    counter_ad += 1
    print()
    num_files = len(listdir(path))
    print(str(num_files) + " images in total in this folder")
    print()
    print(str(counter_ad/num_files) + " percent of the data has AD label")
    print(str(counter_cn/num_files) + " percent of the data has CN label")
    print(str(counter_mci/num_files) + " percent of the data has MCI label")

#%%   
# define a function to do random check with the preprocessed images
def check(path):
    names = listdir(path)
    idx = random.sample(range(0, len(names)), 1)[0]
    img = np.load(path + names[idx])
    print(names[idx])
    # check shape is correct
    print(img.shape) 
    # check range of pixel values
    img_flat = img.flatten()
    print(img_flat[np.argmax(img_flat)])
    print(img_flat[np.argmin(img_flat)])   
    # view in in 3D
    multi_slice_viewer(img, [0, 600, 0, 600])  
    