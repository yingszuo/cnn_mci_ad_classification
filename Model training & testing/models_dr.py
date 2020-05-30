# Author: Ying Zuo 

"""
Model training and testing - Dimension reduction

"""

# import packages
import logging
import numpy as np
import tensorflow as tf
from google.colab import drive

# import all the defined functions for model training & testing
from functions_model.py import *


#%%

"""  Load in the input -  original data"""

Xtr = np.load("inputs/allyr_64/train_data.npy", allow_pickle = True)
ytr = np.load("inputs/allyr_64/train_label.npy", allow_pickle = True)
print(Xtr.shape)
print(ytr.shape)

Xts = np.load("inputs/allyr_64/test_data.npy", allow_pickle = True)
yts = np.load("inputs/allyr_64/test_label.npy", allow_pickle = True)
print(Xts.shape)
print(yts.shape)

Xval = np.load("inputs/allyr_64/val_data.npy", allow_pickle = True)
yval = np.load("inputs/allyr_64/val_label.npy", allow_pickle = True)
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

""" Dimenstion reduction"""

# reshape the input
X_train = Xtr.reshape(-1,64,64,64,1) 
X_test = Xts.reshape(-1,64,64,64,1) 
X_val = Xval.reshape(-1,64,64,64,1) 

# traninng convolutional autoencoder(CAE)
cae = run_cae(X_train, X_val)

# use CAE to predict the train, validation and test data
encoder = Model(cae.input, cae.get_layer('bottleneck').output)

# predict
Xtr_dr = encoder.predict(X_train)
Xval_dr = encoder.predict(X_val)
Xts_dr = encoder.predict(X_test)

# check the new input shape
print(Xtr_dr.shape)
print(Xval_dr.shape)
print(Xts_dr.shape)

#%%


"""  NC vs. AD - dimension reduction """

# create input for binary classification of NC vs. AD
Xtr_ncad, ytr_ncad = data_filter(Xtr_dr, ytr, 1)
Xval_ncad, yval_ncad = data_filter(Xval_dr, yval, 1)
Xts_ncad, yts_ncad = data_filter(Xts_dr, yts, 1)

# rename the input
X_train = Xtr_ncad
X_test = Xts_ncad
X_val = Xval_ncad

# one hot encode the target labels 
y_train = onehot_bi(ytr_ncad)
y_test = onehot_bi(yts_ncad)
y_val = onehot_bi(yval_ncad)


# model training
model, hist = run_adpbase(X_train, y_train, X_val, y_val, lr = 0.001)

# visualization
history_dict = hist.history
#print(history_dict.keys())
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plot_history(data_list=[loss, val_loss],
             label_list=['Training loss', 'Validation loss'],
             title='Training and validation loss',
             ylabel='Loss', name = 'dr_ncad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'dr_ncad_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_adpbase(X_train_ms, y_train_ms, lr = 0.001, final = True)



# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'dr_roc_ncad')

#%%
 

"""  NC vs. MCI - dimension reduction """

# create input for binary classification of NC vs. MCI
Xtr_ncmci, ytr_ncmci = data_filter(Xtr_dr, ytr, 2)
Xval_ncmci, yval_ncmci = data_filter(Xval_dr, yval, 2)
Xts_ncmci, yts_ncmci = data_filter(Xts_dr, yts, 2)

# rename the input
X_train = Xtr_ncmci
X_test = Xts_ncmci
X_val = Xval_ncmci

# one hot encode the target labels 
y_train = onehot_bi(ytr_ncmci)
y_test = onehot_bi(yts_ncmci)
y_val = onehot_bi(yval_ncmci)


# model training
model, hist = run_adpbase(X_train, y_train, X_val, y_val)


# visualization
history_dict = hist.history
#print(history_dict.keys())
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plot_history(data_list=[loss, val_loss],
             label_list=['Training loss', 'Validation loss'],
             title='Training and validation loss',
             ylabel='Loss', name = 'dr_ncmci_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'dr_base_ncmci_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_adpbase(X_train_ms, y_train_ms, final = True)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'dr_roc_ncmci')

#%%


"""  MCI vs. AD - dimension reduction """

# create input for binary classification of MCI vs. AD
Xtr_admci, ytr_admci = data_filter(Xtr_dr, ytr, 0)
Xval_admci, yval_admci = data_filter(Xval_dr, yval, 0)
Xts_admci, yts_admci = data_filter(Xts_dr, yts, 0)

# reshape the input
X_train = Xtr_admci
X_test = Xts_admci
X_val = Xval_admci

# one hot encode the target labels 
y_train = onehot_bi(ytr_admci)
y_test = onehot_bi(yts_admci)
y_val = onehot_bi(yval_admci)


# model training
model, hist = run_adpbase(X_train, y_train, X_val, y_val)


# visualization
history_dict = hist.history
#print(history_dict.keys())
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plot_history(data_list=[loss, val_loss],
             label_list=['Training loss', 'Validation loss'],
             title='Training and validation loss',
             ylabel='Loss', name = 'dr_mciad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'dr_mciad_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_adpbase(X_train_ms, y_train_ms, final = True)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'dr_roc_mciad')

#%%


"""  NC vs. MCI vs. AD - dimension reduction """

# reshape the input for 3-way classification of NC vs. AD
X_train = Xtr_dr
X_test = Xts_dr
X_val = Xval_dr

# one hot encode the target labels 
y_train = onehot_tri(ytr)
y_test = onehot_tri(yts)
y_val = onehot_tri(yval)


# model training
model, hist = run_adpbase(X_train, y_train, X_val, y_val, dr = 0.1, out = 3)

# visualization
history_dict = hist.history
#print(history_dict.keys())
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

plot_history(data_list=[loss, val_loss],
             label_list=['Training loss', 'Validation loss'],
             title='Training and validation loss',
             ylabel='Loss', name = 'dr_ncmciad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'dr_ncmciad_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_adpbase(X_train_ms, y_train_ms, dr = 0.1, out = 3, final = True)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_3way(X_test, y_test, model)




