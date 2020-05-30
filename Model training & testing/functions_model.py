# Author: Ying Zuo 

"""
functions for model training and testing. 
Adapted Gensis is not included in the code below 
(it is incuded in model_training_transfer_learning), 
as it requires a different version of tensorflow. 

"""
#%%
# import packages 
import numpy as np
import matplotlib.pyplot as plt
from imblearn.metrics import sensitivity_specificity_support as sss
from sklearn.metrics import  f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix as CM

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Conv3D, GlobalAveragePooling3D,  MaxPooling3D, LeakyReLU, BatchNormalization, Dropout, Flatten, Activation, Reshape,  Conv3DTranspose, UpSampling3D
from tensorflow.keras.regularizers import l2, l1, l1_l2

#%%

# cretae the binary input 
def data_filter(data, label, exclu):
  idx = np.where(label!= exclu)[0]
  #print(len(idx))
  data_new = data[idx]
  label_new = label[idx]
  #print(data_new.shape)
  print(np.unique(label_new, return_counts=True))
  return data_new, label_new

# onehot encode labels for binary classifications
def onehot_bi(y):
  from sklearn.preprocessing import OneHotEncoder
  onehot_encoder = OneHotEncoder(sparse=False)
  y = y.reshape(len(y), 1)
  y_encoded = onehot_encoder.fit_transform(y)
  return y_encoded

# onehot encode labels for 3-way classifications
def onehot_tri(y):
  from keras.utils import to_categorical
  return to_categorical(y)

# view the distribution of class labels of the input data
def showpercentage(array):
    pcn = array[1][0]/np.sum(array[1])
    pmci = array[1][1]/np.sum(array[1])
    pad = array[1][2]/np.sum(array[1])
    print(str(pad) + " percent of the data has AD label")
    print(str(pcn) + " percent of the data has CN label")
    print(str(pmci) + " percent of the data has MCI label")

#%%
# models

# build the baseline model
def run_base(X_train, y_train, X_valid = None, y_valid = None, 
             final = False, out = 2,
             dr = 0.02, lr = 0.00001, 
             breg = l2(0.0001), areg = None, 
             n_epochs = 30, batch_size = 15):
  
  dim = (64, 64, 64, 1)
  
  model = Sequential()
  model.add(Conv3D(32, kernel_size=(5,5,5),  kernel_initializer='he_uniform', bias_regularizer=breg, input_shape=dim))
  #model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling3D(pool_size=(2, 2, 2)))

  model.add(Conv3D(64, kernel_size=(5,5,5),  bias_regularizer=breg, kernel_initializer='he_uniform'))
  #model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling3D(pool_size=(2,2,2)))


  model.add(Conv3D(128, kernel_size=(5,5,5),  bias_regularizer=breg, kernel_initializer='he_uniform'))
  #model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling3D(pool_size=(2,2,2)))

  model.add(Dropout(dr))

  model.add(Flatten())
  model.add(Dense(512, bias_regularizer=breg,   kernel_initializer='he_uniform'))
  #model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Dropout(dr))

  model.add(Dense(256, bias_regularizer=breg,   kernel_initializer='he_uniform'))
  #model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Dense(out, activation='softmax', activity_regularizer=areg))

  # model optimization
  opt = Adam(learning_rate = lr)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) 

  cb = ReduceLROnPlateau(monitor = 'val_loss', 
                         factor = 0.5, patience = 5, 
                         verbose = 1, epsilon = 1e-4, mode = 'min')
    
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


# build the convolutional autoencoder
def run_cae(X_train,  X_val, lr = 0.0001, 
        n_epochs = 100, batch_size = 10):
  
  dim = (64,64,64,1)
  
  inp = Input(dim)
  # Encoder
  e = Conv3D(32, (3, 3, 3), strides = 2, activation='elu',  kernel_initializer='he_uniform', padding = "same")(inp)
  e = Conv3D(64, (3, 3, 3), strides = 2, activation='elu',  kernel_initializer='he_uniform', padding = "same")(e)
  e = Conv3D(1, (3, 3, 3), strides = 2,activation='elu',  kernel_initializer='he_uniform', padding = "same", name = "bottleneck")(e)

  #DECODER
  d = Conv3DTranspose(64,(3,3,3), strides = 2, kernel_initializer='he_uniform', activation='elu', padding = "same")(e)
  d = BatchNormalization()(d)
  d = Conv3DTranspose(16,(3,3,3), strides=2,  kernel_initializer='he_uniform', activation='elu', padding = "same")(d)
  d = BatchNormalization()(d)
  d = Conv3DTranspose(16,(3,3,3), strides=2,  kernel_initializer='he_uniform', activation='elu', padding = "same")(d)
  decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(d)

  ae = Model(inp, decoded)

  # model optimization
  opt_ae = Adam(learning_rate = lr)
  ae.compile(optimizer = opt_ae, loss = "mse")
  cb = ReduceLROnPlateau(monitor = 'val_loss', 
                         factor = 0.9, patience = 3, 
                         verbose = 1, mode = 'min')

  #Train it by providing training images
  ae.fit(X_train, X_train, 
         batch_size = batch_size, 
         epochs = n_epochs, 
         validation_data = (X_val, X_val), 
         verbose = 1,
         callbacks = [cb])
  
  return ae
  

# build the adapted baseline model for training with dimension reduced data
def run_adpbase(X_train, y_train, X_valid = None, y_valid = None, 
             final = False, out = 2,
             dr = 0.2, lr = 0.0001, 
             breg = l2(0.00001), wreg = l2(0.00001), areg = l1(0.00001), 
             n_epochs = 100, batch_size = 25):
  
  dim = (8,8,8,1)
   
  model = Sequential()
  model.add(Conv3D(32, kernel_size=(2,2,2), kernel_initializer='he_uniform',bias_regularizer=breg, kernel_regularizer=wreg,input_shape=dim))
  model.add(BatchNormalization())
  model.add(Activation('relu'))

  model.add(Conv3D(64, kernel_size=(2,2,2), kernel_initializer='he_uniform',bias_regularizer=breg, kernel_regularizer=wreg,input_shape=dim))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling3D(pool_size=(2,2,2)))
  model.add(Dropout(dr))

  model.add(Conv3D(128, kernel_size=(2,2,2), kernel_initializer='he_uniform',bias_regularizer=breg, kernel_regularizer=wreg,input_shape=dim))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling3D(pool_size=(2,2,2)))
  model.add(Dropout(dr))

  model.add(Flatten())
  model.add(Dense(512,kernel_initializer='he_uniform', kernel_regularizer=wreg,bias_regularizer=breg))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(dr))

  model.add(Dense(out, activation='softmax', activity_regularizer=areg))

  # model optimization
  opt = Adam(learning_rate=lr)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) 
  cb = ReduceLROnPlateau(monitor = 'val_loss', 
                         factor = 0.5, patience = 5, 
                         verbose = 1, epsilon = 1e-4, mode = 'min')
    
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
              callbacks = [cb],
              shuffle = True)


  return model, hist




#%%
  
# visualizatio of model traning and model performance 

# visualize the training and validation performance
def plot_history(data_list, label_list, title, ylabel, name):

    epochs = range(1, len(data_list[0]) + 1)

    for data, label in zip(data_list, label_list):
        plt.plot(epochs, data, label=label)
    plt.title(title, pad = 10, fontsize='large')
    plt.xlabel('Epochs', labelpad=10)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
#%%

# model evaluation
    
# evaluate model performance - binary classifications
def evaluate_binary(X_test, y_test, model, name):
  test_y_prob = model.predict(X_test)
  test_y_pred = np.argmax(test_y_prob, axis = 1)
  test_y_true = np.argmax(y_test, axis = 1) 
  # accuracy
  loss, acc = model.evaluate(X_test, y_test)
  # AUC
  pos_prob = test_y_prob[:,1]
  auc_score = roc_auc_score(test_y_true, pos_prob)
  # precision, recall, specificity, and f1_score
  p = precision_score(test_y_true, test_y_pred)
  r = recall_score(test_y_true, test_y_pred)
  f1 = f1_score(test_y_true, test_y_pred)
  sen, spe, _ = sss(test_y_true, test_y_pred, average="binary")
  
  # print results
  print("Test accuracy:", acc)
  print("Test AUC is: ", auc_score)
  print("Test confusion matrix: \n", CM(test_y_true, test_y_pred))
  print("Precision: ", p)
  print("Recall: ", r)
  print("Specificity: ", spe)
  print("f1_score: ", f1)
  
  # plot and save roc curve
  pos_prob = test_y_prob[:,1]
  fpr, tpr, thresholds = roc_curve(test_y_true, pos_prob)
  ns_probs = [0 for _ in range(len(test_y_prob))]
  ns_fpr, ns_tpr, _ = roc_curve(test_y_true, ns_probs)
  plt.axis([0,1,0,1]) 
  plt.plot(fpr,tpr, marker = '.', color = 'darkorange', label = 'Model AUC (area = {:.2f})'.format(auc_score)) 
  plt.plot(ns_fpr, ns_tpr, color = 'royalblue', linestyle='--')
  plt.legend()
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.savefig(name, dpi=300, bbox_inches='tight')
  plt.show()


# evaluate model performance - 3 way classifiations
def evaluate_3way(X_test, y_test, model):
  test_y_prob = model.predict(X_test)
  test_y_pred = np.argmax(test_y_prob, axis = 1)
  test_y_true = np.argmax(y_test, axis = 1) 
  # accuracy
  loss, acc = model.evaluate(X_test, y_test)
  # precision, recall, specificity, and f1_score
  p = precision_score(test_y_true, test_y_pred, average="macro")
  r = recall_score(test_y_true, test_y_pred, average="macro")
  f1 = f1_score(test_y_true, test_y_pred, average="macro")
  sen,spe,_ = sss(test_y_true, test_y_pred, average="macro")

  print("Test accuracy:", acc)
  print("Test confusion matrix: \n", CM(test_y_true, test_y_pred))
  print("Precision: ", p)
  print("Recall: ", r)
  print("Specificity: ", spe)
  print("f1_score: ", f1)

