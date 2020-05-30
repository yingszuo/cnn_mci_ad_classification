# Author: Ying Zuo 


"""
Model training and testing - Baseline

"""

# import packages
import logging
import numpy as np
import tensorflow as tf
from google.colab import drive

# import all the defined functions for model training & testing
from functions_model.py import *


#%% 


"""  Load in the input - original data """

Xtr = np.load("inputs/allyr_64/train_data.npy", allow_pickle = True)
ytr = np.load("inputs/allyr_64/train_label.npy", allow_pickle = True)
#print(Xtr.shape)
#print(ytr.shape)

Xts = np.load("inputs/allyr_64/test_data.npy", allow_pickle = True)
yts = np.load("inputs/allyr_64/test_label.npy", allow_pickle = True)
#print(Xts.shape)
#print(yts.shape)

Xval = np.load("inputs/allyr_64/val_data.npy", allow_pickle = True)
yval = np.load("inputs/allyr_64/val_label.npy", allow_pickle = True)
#print(Xval.shape)
#print(yval.shape)

print("Train:")
showpercentage(np.unique(ytr, return_counts=True))
print()
print("Validation:")
showpercentage(np.unique(yval, return_counts=True))
print()
print("Test")
showpercentage(np.unique(yts, return_counts=True))

#%% 


""" Baseline NC vs. AD """

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



# model training
model, hist = run_base(X_train, y_train, X_val, y_val, 
                       breg = l2(0.001), areg = l1(0.001))


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
             ylabel='Loss', name = 'base_ncad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'base_ncad_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
#print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
#print(y_train_ms.shape)

model, _  = run_base(X_train_ms, y_train_ms, 
                     breg = l2(0.001), areg = l1(0.001), final = True)



# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'base_roc_ncad')

#%% 


"""  Baseline NC vs. MCI """

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


# model training
model, hist = run_base(X_train, y_train, X_val, y_val, areg = l1(0.001))

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
             ylabel='Loss', name = 'base_ncmci_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'base_ncmci_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
#print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
#print(y_train_ms.shape)

model, _  = run_base(X_train_ms, y_train_ms, areg = l1(0.001), final = True)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'base_roc_ncmci')

#%% 


"""  Baseline MCI vs. AD """

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


# model training
model, hist = run_base(X_train, y_train, X_val, y_val, dr = 0.03)


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
             ylabel='Loss', name = 'base_mciad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'base_mciad_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
#print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
#print(y_train_ms.shape)

model, _  = run_base(X_train_ms, y_train_ms, dr = 0.03, final = True)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'base_roc_mciad')

#%% 


"""   Baseline NC vs. MCI vs. AD """

# reshape the input for 3-way classification of NC vs. AD
X_train = Xtr.reshape(-1,64,64,64,1) 
X_test = Xts.reshape(-1,64,64,64,1) 
X_val = Xval.reshape(-1,64,64,64,1) 

# one hot encode the target labels 
y_train = onehot_tri(ytr)
y_test = onehot_tri(yts)
y_val = onehot_tri(yval)


# model training
model, hist = run_base(X_train, y_train, X_val, y_val, 
                       breg = l2(0.001), out = 3)


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
             ylabel='Loss', name = 'base_ncmciad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'base_ncmciad_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
#print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
#print(y_train_ms.shape)

model, _  = run_base(X_train_ms, y_train_ms, 
                     breg = l2(0.001), out = 3, final = True)



# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_3way(X_test, y_test, model)


##############################  model performance with randomly split data  ####################################3
#%%


"""# Load in the input -  randomly split data"""

Xtr_rs = np.load("inputs/rs_allyr_64/train_data.npy", allow_pickle = True)
ytr_rs = np.load("inputs/rs_allyr_64/train_label.npy", allow_pickle = True)

Xts_rs = np.load("inputs/allyr_64/test_data.npy", allow_pickle = True)
yts_rs = np.load("inputs/allyr_64/test_label.npy", allow_pickle = True)


Xval_rs = np.load("inputs/rs_allyr_64/val_data.npy", allow_pickle = True)
yval_rs = np.load("inputs/rs_allyr_64/val_label.npy", allow_pickle = True)

#%%
 

"""  Randomly split data - NC vs. AD """

# create input for binary classification of NC vs. AD
Xtr_ncad, ytr_ncad = data_filter(Xtr_rs, ytr_rs, 1)
Xval_ncad, yval_ncad = data_filter(Xval_rs, yval_rs, 1)
Xts_ncad, yts_ncad = data_filter(Xts_rs, yts_rs, 1)

# reshape the input
X_train = Xtr_ncad.reshape(-1,64,64,64,1) 
X_test = Xts_ncad.reshape(-1,64,64,64,1) 
X_val = Xval_ncad.reshape(-1,64,64,64,1) 

# one hot encode the target labels 
y_train = onehot_bi(ytr_ncad)
y_test = onehot_bi(yts_ncad)
y_val = onehot_bi(yval_ncad)


# model training
model, hist = run_base(X_train, y_train, X_val, y_val, n_epochs = 20)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'base_rs_roc_ncad')

#%%


"""  Randomly split data - NC vs. MCI """

# create input for binary classification of NC vs. MCI
Xtr_ncmci, ytr_ncmci = data_filter(Xtr_rs, ytr_rs, 2)
Xval_ncmci, yval_ncmci = data_filter(Xval_rs, yval_rs, 2)
Xts_ncmci, yts_ncmci = data_filter(Xts_rs, yts_rs, 2)

# reshape the input
X_train = Xtr_ncmci.reshape(-1,64,64,64,1) 
X_test = Xts_ncmci.reshape(-1,64,64,64,1) 
X_val = Xval_ncmci.reshape(-1,64,64,64,1) 

# one hot encode the target labels 
y_train = onehot_bi(ytr_ncmci)
y_test = onehot_bi(yts_ncmci)
y_val = onehot_bi(yval_ncmci)


# model training
model, hist = run_base(X_train, y_train, X_val, y_val, n_epochs = 20)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'base_rs_roc_ncmci')

#%%


"""  Randomly split data - MCI vs. AD """

# create input for binary classification of NC vs. MCI
Xtr_admci, ytr_admci = data_filter(Xtr_rs, ytr_rs, 0)
Xval_admci, yval_admci = data_filter(Xval_rs, yval_rs, 0)
Xts_admci, yts_admci = data_filter(Xts_rs, yts_rs, 0)

# reshape the input
X_train = Xtr_admci.reshape(-1,64,64,64,1) 
X_test = Xts_admci.reshape(-1,64,64,64,1) 
X_val = Xval_admci.reshape(-1,64,64,64,1) 

# one hot encode the target labels 
y_train = onehot_bi(ytr_admci)
y_test = onehot_bi(yts_admci)
y_val = onehot_bi(yval_admci)


# model training
model, hist = run_base(X_train, y_train, X_val, y_val, n_epochs = 20)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'base_rs_roc_mciad')

#%%


"""  Randomly split data - NC vs. MCI vs. AD"""

# reshape the input for 3-way classification of NC vs. MCI vs. AD
X_train = Xtr_rs.reshape(-1,64,64,64,1) 
X_test = Xts_rs.reshape(-1,64,64,64,1) 
X_val = Xval_rs.reshape(-1,64,64,64,1) 


# one hot encode the target labels 
y_train = onehot_tri(ytr_rs)
y_test = onehot_tri(yts_rs)
y_val = onehot_tri(yval_rs)


# model training
model, hist = run_base(X_train, y_train, X_val, y_val, out = 3, n_epochs = 20)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_3way(X_test, y_test, model)








