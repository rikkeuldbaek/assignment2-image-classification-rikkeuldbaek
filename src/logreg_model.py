
### Visual Assignment 2
## Cultural Data Science - Visual Analytics 
# Author: Rikke Uldbæk (202007501)
# Date: 14th of March 2023

#--------------------------------------------------------#
############ LOGISTIC REGRESSION CLASSIFIER #############
#--------------------------------------------------------#

# (please note that some of this code has been adapted from class sessions)

# Import packages
# Path tools 
import os

# Data munging tools
import numpy as np
import cv2

# Data loader and saver
from numpy import loadtxt
from joblib import dump, load

# machine learning tools
from sklearn.metrics import classification_report

# Classification model
from sklearn.linear_model import LogisticRegression

# Scripting
import argparse


############# Parser function #############

def input_parse():
    #initialise the parser
    parser = argparse.ArgumentParser()
    #add arguments
    parser.add_argument("--penalty", type= str, default="none", help= "Specify norm of penalty.")
    parser.add_argument("--tol", type= float, default=0.1, help= "Specify tolerance for stopping criteria.")
    parser.add_argument("--verbose", type= bool, default= False, help= "Specify whether model should show output.")
    parser.add_argument("--solver", type= str, default="saga", help= "Specify solver algorithm for optimization problem.")
    parser.add_argument("--multi_class", type= str, default="multinomial", help= "Specify multi class.")
    # parse the arguments from the command line 
    args = parser.parse_args()
    #define a return value
    return(args) #returning arguments

# Fetching data
import data as dt
X_train, y_train, X_test, y_test, labels = dt.loading_data()
X_train_grey , X_test_grey = dt.greyscale(X_train, X_test)
X_train_scaled, X_test_scaled = dt.scale(X_train_grey , X_test_grey)
X_train_dataset, X_test_dataset= dt.reshape(X_train_scaled, X_test_scaled)

######### LOGISTIC REGRESSION MODEL ############

def logreg_model_function(penalty, tol, verbose, solver, multi_class):

    print("Initializing logistic regression classification..")

    # Logistic regression model
    clf = LogisticRegression(penalty= penalty, 
                        tol= tol,
                        verbose= verbose,
                        solver= solver,
                        multi_class= multi_class).fit(X_train_dataset, y_train)

    print("Norm of the penalty =", penalty)
    print("Tolerance for stopping criteria =", tol)
    print("Verbose =", verbose)
    print("Solver algorithm for optimization problem =", solver)
    print("Labels to fit =", multi_class)

    # Predict
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
    outpath_metrics_report = os.path.join(os.getcwd(), "out", "LR_metrics_report.txt")

    # Save the metrics report
    file = open(outpath_metrics_report, "w")
    file.write(report)
    file.close()

    # Save the trained model to the folder called "models"
    # Define out path
    outpath_classifier = os.path.join(os.getcwd(), "models", "LR_classifier.joblib")

    # Save model
    dump(clf, open(outpath_classifier, 'wb'))

    print( "Saving the logistic regression metrics report in the folder ´out´")
    print( "Saving the logistic regression model in the folder ´models´")



############# Main function #############
def main():
    # input parse
    args = input_parse()
    # pass arguments to logistic regression function
    report, clf = logreg_model_function(args.penalty, args.tol, args.verbose, args.solver, args.multi_class)
    save_results(report, clf)

main()