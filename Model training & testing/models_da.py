# Author: Ying Zuo 


"""
Model training and testing - Data augmentation

"""

# import packages
import logging
import numpy as np
import tensorflow as tf
from google.colab import drive

# import all the defined functions for model training & testing
from functions_model.py import *


#%%
"""# Load in the augmented training data"""

Xtr = np.load("inputs/aug_data/train_data.npy", allow_pickle = True)
ytr = np.load("inputs/aug_data/train_label.npy", allow_pickle = True)
print(Xtr.shape)
print(ytr.shape)

Xts = np.load("inputs/aug_data/test_data.npy", allow_pickle = True)
yts = np.load("inputs/aug_data/test_label.npy", allow_pickle = True)
print(Xts.shape)
print(yts.shape)

Xval = np.load("inputs/aug_data/val_data.npy", allow_pickle = True)
yval = np.load("inputs/aug_data/val_label.npy", allow_pickle = True)
print(Xval.shape)
print(yval.shape)

print("Train:")
showpercentage(np.unique(ytr, return_counts=True))
print()
print("Validation:")
showpercentage(np.unique(yval, return_counts=True))
print()
print("Test")
showpercentage(np.unique(yts, return_counts=True))

#%%
#  NC vs. AD - data augmentation

"""# Preparing data & labels for NC vs. AD"""

# create input for binary classification of NC vs. AD
Xtr_ncad, ytr_ncad = data_filter(Xtr, ytr, 1)
Xval_ncad, yval_ncad = data_filter(Xval, yval, 1)
Xts_ncad, yts_ncad = data_filter(Xts, yts, 1)

# reshape the input
X_train = Xtr_ncad.reshape(-1,64,64,64,1) 
X_test = Xts_ncad.reshape(-1,64,64,64,1) 
X_val = Xval_ncad.reshape(-1,64,64,64,1) 

# one hot encode the target labels 
y_train = onehot_bi(ytr_ncad)
y_test = onehot_bi(yts_ncad)
y_val = onehot_bi(yval_ncad)


"""# Model training - NC vs. AD"""

model, hist = run_base(X_train, y_train, X_val, y_val)

"""# Visualization of model training -  NC vs. AD"""

history_dict = hist.history
#print(history_dict.keys())
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# plotting training and validation loss and accuracy
plot_history(data_list=[loss, val_loss],
             label_list=['Training loss', 'Validation loss'],
             title='Training and validation loss',
             ylabel='Loss', name = 'da_ncad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'da_ncad_acc')

"""# Model final training - NC vs. AD"""

X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_base(X_train_ms, y_train_ms, final = True)

"""# Model testing - NC vs. AD"""

# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'da_roc_ncad')

#%%
# 
"""# Preparing data & labels for NC vs. MCI"""
#  NC vs. MCI - data augmentation

# create input for binary classification of NC vs. MCI
Xtr_ncmci, ytr_ncmci = data_filter(Xtr, ytr, 2)
Xval_ncmci, yval_ncmci = data_filter(Xval, yval, 2)
Xts_ncmci, yts_ncmci = data_filter(Xts, yts, 2)

# reshape the input
X_train = Xtr_ncmci.reshape(-1,64,64,64,1) 
X_test = Xts_ncmci.reshape(-1,64,64,64,1) 
X_val = Xval_ncmci.reshape(-1,64,64,64,1) 

# one hot encode the target labels 
y_train = onehot_bi(ytr_ncmci)
y_test = onehot_bi(yts_ncmci)
y_val = onehot_bi(yval_ncmci)


"""# Model training - NC vs. MCI"""

model, hist = run_base(X_train, y_train, X_val, y_val, dr = 0.04)

"""# Visualization of model training - NC vs. MCI"""

history_dict = hist.history
#print(history_dict.keys())
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# plotting training and validation loss and accuracy
plot_history(data_list=[loss, val_loss],
             label_list=['Training loss', 'Validation loss'],
             title='Training and validation loss',
             ylabel='Loss', name = 'da_ncmci_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'da_ncmci_acc')

"""# Model final training - NC vs. MCI"""

X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_base(X_train_ms, y_train_ms, dr = 0.04, final = True)

"""# Model testing - NC vs. MCI"""

# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'da_roc_ncmci')

#%%
#  MCI vs. AD - data augmentation

"""# Preparing data & labels for MCI vs. AD"""

# create input for binary classification of MCI vs. AD
Xtr_admci, ytr_admci = data_filter(Xtr, ytr, 0)
Xval_admci, yval_admci = data_filter(Xval, yval, 0)
Xts_admci, yts_admci = data_filter(Xts, yts, 0)

# reshape the input
X_train = Xtr_admci.reshape(-1,64,64,64,1) 
X_test = Xts_admci.reshape(-1,64,64,64,1) 
X_val = Xval_admci.reshape(-1,64,64,64,1) 

# one hot encode the target labels 
y_train = onehot_bi(ytr_admci)
y_test = onehot_bi(yts_admci)
y_val = onehot_bi(yval_admci)


"""# Model training - MCI vs. AD"""

model, hist = run_base(X_train, y_train, X_val, y_val, 
                       dr = 0.05, breg = l2(0.001), areg = l1(0.0001))

"""# Visualization of model training - MCI vs. AD"""

history_dict = hist.history
#print(history_dict.keys())
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# plotting training and validation loss and accuracy
plot_history(data_list=[loss, val_loss],
             label_list=['Training loss', 'Validation loss'],
             title='Training and validation loss',
             ylabel='Loss', name = 'da_mciad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'da_mciad_acc')

"""# Model final training - MCI vs. AD"""

X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_base(X_train_ms, y_train_ms, 
                     dr = 0.05, breg = l2(0.001), areg = l1(0.0001), 
                     final = True)

"""# Model testing - MCI vs. AD"""

# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'da_roc_mciad')

#%% 
# NC vs. MCI vs. AD - data augmentation

"""# Preparing data & labels for NC vs. MCI vs. AD"""

# reshape the input for 3-way classification of NC vs. AD
X_train = Xtr.reshape(-1,64,64,64,1) 
X_test = Xts.reshape(-1,64,64,64,1) 
X_val = Xval.reshape(-1,64,64,64,1) 

# one hot encode the target labels 
y_train = onehot_tri(ytr)
y_test = onehot_tri(yts)
y_val = onehot_tri(yval)


"""# Model training - NC vs. MCI vs. AD"""

model, hist = run_base(X_train, y_train, X_val, y_val, 
                       breg = l2(0.001), out = 3)

"""# Visualization of model training - NC vs. MCI vs. AD"""

history_dict = hist.history
#print(history_dict.keys())
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# plotting training and validation loss and accuracy
plot_history(data_list=[loss, val_loss],
             label_list=['Training loss', 'Validation loss'],
             title='Training and validation loss',
             ylabel='Loss', name = 'da_ncmciad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'da_ncmciad_acc')

"""# Model final training - NC vs. MCI vs. AD"""

X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_base(X_train_ms, y_train_ms, 
                     breg = l2(0.001), out = 3, final = True)

"""# Model testing - NC vs. MCI vs. AD"""

# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_3way(X_test, y_test, model)