### Visual Assignment 2
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 14th of March 2023

#--------------------------------------------------------#
################## DATA PREPROCESSING ####################
#--------------------------------------------------------#
 
# (please note that some of this code has been adapted from class sessions)

# import packages
# Path tools
import os

# Data munging tools
import numpy as np
import cv2

# Data loader and saver
from tensorflow.keras.datasets import cifar10
from numpy import savetxt



######### DATA PREPROCESSING ############

print("Initializing data preprocessing..")

# Read in the data
def loading_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    labels = ["airplane", 
        "automobile", 
        "bird", 
        "cat", 
        "deer", 
        "dog", 
        "frog", 
        "horse", 
        "ship", 
        "truck"] #note this is alfabetically 
    return X_train, y_train, X_test, y_test, labels


# Convert to greyscale
def greyscale(X_train, X_test):
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    return X_train_grey , X_test_grey 

# Scaling
def scale(X_train_grey , X_test_grey):
    X_train_scaled = (X_train_grey)/255.0 
    X_test_scaled = (X_test_grey)/255.0 
    return  X_train_scaled, X_test_scaled 

# Reshaping
def reshape(X_train_scaled, X_test_scaled):
    # Reshape training data 
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    # Reshape test data 
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))
    return X_train_dataset, X_test_dataset

print("Data preprocessing done!")

def main():
    loading_data()
    greyscale(X_train, X_test)
    scale(X_train_grey , X_test_grey)
    reshape(X_train_scaled, X_test_scaled)

if __name__ == '__main__':
    main()

