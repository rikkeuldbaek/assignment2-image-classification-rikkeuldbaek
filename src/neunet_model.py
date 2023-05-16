### Visual Assignment 2
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 14th of March 2023

#--------------------------------------------------------#
############### NEURAL NETWORK CLASSIFIER ###############
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

# Import packages
# path tools
import os
import sys
sys.path.append(os.path.join(".."))

# Data munging tools
import cv2
import numpy as np

# Data loader and saver
from numpy import loadtxt
from joblib import dump, load

# Machine learning tools
from sklearn.metrics import classification_report

# classification model
from sklearn.neural_network import MLPClassifier

# Scripting
import argparse


############# Parser function #############

def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()
    #add arguments
    parser.add_argument("--random_state", type= int, default= 666, help="Specify random state.")
    parser.add_argument("--hidden_layer_sizes", nargs = "+", type= int, default = (64,20), help="Specify hidden layer sizes. More hidden layers will increase computational time. This must be specified without commas.")
    parser.add_argument("--learning_rate", type = str, default= "adaptive", help= "Specify learning rate.") 
    parser.add_argument("--early_stopping", type= bool, default= True, help = "Specify early stopping.")
    parser.add_argument("--verbose", type= bool, default= True, help = "Specify whether model should show output.")
    parser.add_argument("--max_iter", type= int, default=20, help = "Specify maximum number of iterations.")

    args = parser.parse_args()
    return(args) #returning arguments


######### IMPORTING DATA ############
import data as dt
X_train, y_train, X_test, y_test, labels = dt.loading_data()
X_train_grey , X_test_grey = dt.greyscale(X_train, X_test)
X_train_scaled, X_test_scaled = dt.scale(X_train_grey , X_test_grey)
X_train_dataset, X_test_dataset= dt.reshape(X_train_scaled, X_test_scaled)


######### NEURAL NETWORK MODEL ############
def neural_network_function(random_state, hidden_layer_sizes, learning_rate, early_stopping, verbose, max_iter):

    print("Initializing neural network classification..")    

    # Neural network model
    clf = MLPClassifier(random_state= random_state,
                        hidden_layer_sizes= hidden_layer_sizes,
                        learning_rate= learning_rate, 
                        early_stopping= early_stopping,
                        verbose=verbose,
                        max_iter= max_iter).fit(X_train_dataset, y_train)

    print("Hidden layer sizes =", hidden_layer_sizes )
    print("Learning rate schedule for weight updates =", learning_rate)
    print("Early stopping of training when validation score is not improving =", early_stopping)
    print("Verbose =", verbose)
    print("Maximum number of iterations =", max_iter)

    # Prediction
    y_pred = clf.predict(X_test_dataset)
    
    # Logistic regression classification report 
    report = classification_report(y_test, 
                               y_pred, 
                               target_names=labels)
    
    return(report, clf)

############# Save model and metrics report #############
def save_results(report, clf):
    # Save the classification report in the folder "out"
    # Define out path
    outpath_metrics_report = os.path.join(os.getcwd(), "out", "NN_metrics_report.txt")

    # Save the metrics report
    file = open(outpath_metrics_report, "w")
    file.write(report)
    file.close()

    # Save the trained model to the folder called "models"
    # Define out path
    outpath_classifier = os.path.join(os.getcwd(), "models", "NN_classifier.joblib")

    # Save model
    dump(clf, open(outpath_classifier, 'wb'))

    print( "Saving the neural network metrics report in the folder ´out´")
    print( "Saving the neural network model in the folder ´models´")



############# Main function #############
def main():
    # input parse
    args = input_parse()
    # pass arguments to Neural Network function
    report, clf = neural_network_function(args.random_state, tuple(args.hidden_layer_sizes), args.learning_rate, args.early_stopping, args.verbose, args.max_iter)
    save_results(report, clf)

if __name__ == '__main__':
    main()

