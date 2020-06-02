# Author: Ying Zuo 


"""
Model training and testing - Transfer learning

"""

# Commented out IPython magic to ensure Python compatibility.
# Sets tensorflow version to 1.x (colab default is 2.x) so that it is compatible
# with the tensorflow version requirements by model genesis
#%tensorflow_version 1.x

# import packages
import os
import logging
import numpy as np
import tensorflow as tf
from google.colab import drive

import keras
from keras.models import * 
from functions_model.py import *
from keras.optimizers import Adam 
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, GlobalAveragePooling3D, Dropout, BatchNormalization

# import all the defined functions for model training & testing
from functions_model.py import *

# import in the pretained model, Genesis
from unet3d import *


#%% 
# build the Genesis model(set %tensorflow_version 1.x for training Genesis)
def run_genesis(X_train, y_train, X_valid = None, y_valid = None, 
             final = False, out = 2,
             dr = 0.02, n_epochs = 30, batch_size = 15):
  
  input_channels, input_rows, input_cols, input_deps = 1, 64, 64, 64

  weight_dir = 'code/MG/keras/pretrained_weights/Genesis_Chest_CT.h5'

  models_genesis = unet_model_3d((input_channels, input_rows, input_cols, input_deps), batch_normalization=True)
  print("Load pre-trained Models Genesis weights from {}".format(weight_dir))
  models_genesis.load_weights(weight_dir)

  x = models_genesis.get_layer('depth_7_relu').output
  x = GlobalAveragePooling3D()(x)
  x = Dense(1024, activation='relu')(x)
  x = Dropout(dr)(x)
  x = Dense(128, activation='relu')(x)
  x = Dropout(dr)(x)

  output = Dense(out, activation = 'softmax')(x)

  # model optimization
  # opt = Adam(learning_rate = lr)
  model = keras.models.Model(inputs=models_genesis.input, outputs=output)
  model.compile(optimizer="adam", loss = 'categorical_crossentropy', metrics=['accuracy'])
  cb = ReduceLROnPlateau(monitor = 'val_loss', 
                         factor = 0.5, patience = 5, 
                         verbose = 1, mode = 'min')
    
  # model training and fine-tuning
  if not final:
    hist = model.fit(X_train, y_train,
                     batch_size = batch_size, 
                     epochs = n_epochs,
                     callbacks=[cb],
                     validation_data = (X_valid, y_valid), 
                     shuffle = True)
  
  # model final training for testing (train + valid combined)
  else:
    hist = model.fit(X_train, y_train,
              batch_size = batch_size, 
              epochs = n_epochs,
              callbacks=[cb],
              shuffle = True)


  return model, hist

#%%
  

"""  Load in the input - original data"""

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


"""  NC vs. AD - transfer learning """

# create input for binary classification of NC vs. AD
Xtr_ncad, ytr_ncad = data_filter(Xtr, ytr, 1)
Xval_ncad, yval_ncad = data_filter(Xval, yval, 1)
Xts_ncad, yts_ncad = data_filter(Xts, yts, 1)

# reshape the input
X_train = np.transpose(Xtr_ncad.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3)) 
X_test = np.transpose(Xts_ncad.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3))
X_val = np.transpose(Xval_ncad.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3))

# one hot encode the target labels 
y_train = onehot_bi(ytr_ncad)
y_test = onehot_bi(yts_ncad)
y_val = onehot_bi(yval_ncad)


# model training
model, hist = run_genesis(X_train, y_train, X_val, y_val)


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
             ylabel='Loss', name = 'tf_ncad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'tf_ncad_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_genesis(X_train_ms, y_train_ms, n_epochs = 60, final = True)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'tf_roc_ncad')

#%%


"""  NC vs. MCI - transfer learning """

# create input for binary classification of NC vs. MCI
Xtr_ncmci, ytr_ncmci = data_filter(Xtr, ytr, 2)
Xval_ncmci, yval_ncmci = data_filter(Xval, yval, 2)
Xts_ncmci, yts_ncmci = data_filter(Xts, yts, 2)

# reshape the input
X_train = np.transpose(Xtr_ncmci.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3)) 
X_test = np.transpose(Xts_ncmci.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3))
X_val = np.transpose(Xval_ncmci.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3))

# one hot encode the target labels 
y_train = onehot_bi(ytr_ncmci)
y_test = onehot_bi(yts_ncmci)
y_val = onehot_bi(yval_ncmci)


# model training
model, hist = run_genesis(X_train, y_train, X_val, y_val)


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
             ylabel='Loss', name = 'tf_ncmci_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'tf_ncmci_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_genesis(X_train_ms, y_train_ms, final = True)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'tf_roc_ncmci')

#%%


"""  MCI vs. AD - transfer learning """

# create input for binary classification of MCI vs. AD
Xtr_admci, ytr_admci = data_filter(Xtr, ytr, 0)
Xval_admci, yval_admci = data_filter(Xval, yval, 0)
Xts_admci, yts_admci = data_filter(Xts, yts, 0)

# reshape the input
X_train = np.transpose(Xtr_admci.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3)) 
X_test = np.transpose(Xts_admci.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3))
X_val = np.transpose(Xval_admci.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3))


# one hot encode the target labels 
y_train = onehot_bi(ytr_admci)
y_test = onehot_bi(yts_admci)
y_val = onehot_bi(yval_admci)

# model training
model, hist = run_genesis(X_train, y_train, X_val, y_val)

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
             ylabel='Loss', name = 'tf_mciad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'tf_mciad_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_genesis(X_train_ms, y_train_ms, final = True)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_binary(X_test, y_test, model, name = 'tf_roc_mciad')

#%% 


"""  NC vs. MCI vs. AD - transfer learning """

# reshape the input for 3-way classification of NC vs. MCI vs. AD
X_train = np.transpose(Xtr.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3)) 
X_test = np.transpose(Xts.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3))
X_val = np.transpose(Xval.reshape(-1,64,64,64,1), (0, 4, 1, 2, 3))

# one hot encode the target labels 
y_train = onehot_tri(ytr)
y_test = onehot_tri(yts)
y_val = onehot_tri(yval)


# model training
model, hist = run_genesis(X_train, y_train, X_val, y_val, out = 3)

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
             ylabel='Loss', name = 'tf_ncmciad_loss')
plot_history(data_list=[acc, val_acc],
             label_list=['Training accuracy', 'Validation accuracy'],
             title ='Training and validation accuracy',
             ylabel ='Accuracy', name = 'tf_ncmciad_acc')


# model final training
X_train_ms = np.concatenate((X_train, X_val), axis = 0)
print(X_train_ms.shape)
y_train_ms = np.concatenate((y_train, y_val), axis = 0)
print(y_train_ms.shape)

model, _  = run_genesis(X_train_ms, y_train_ms, out = 3, final = True)


# evaluate with test set (acc, auc, precision, recall, specificity and f1)
evaluate_3way(X_test, y_test, model)



