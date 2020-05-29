The thesis research focus on investigating using CNNs to classify mild cognitive impairment and alzheimer's disease using 3D brain MRI images. 

There are two functions files: functions_preprocessing.py and functions_model.py. 
The functions_preprocessing file contains predefined functions to preprocess MRI images, and the functions_model file contains functions needed for model training and testing

The image preprocessing procedure is detailed in data_preprocessing.py file, which needs to be used together with functions_preprocessing.py.

Model trainng and testing using four different methods (16 models) is detailed in four files: baseline.py, data_augmentation.py, transfer_learning.py and dimension_reduction.py. Each file contains training and testing of four differnt models using one method. The model training and tesing files need to be used together with functions_model.
