#-------------------------------------------------------------------------
# AUTHOR: Cameron Ross
# FILENAME: svm.py
# SPECIFICATION: Question 4 of Assignment 3.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

from sklearn import svm
import numpy as np
import pandas as pd

# defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) # reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] # getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] # getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) # reading the test data by using Pandas library

X_test = np.array(df.values)[:,:64] # getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] # getting the last field to create the class testing data and convert them to NumPy array

# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
highest_accuracy = 0
for i in range(len(c)):
    for j in range(len(degree)):
        for k in range(len(kernel)):
           for l in range(len(decision_function_shape)):
                # Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                clf = svm.SVC(C=c[i], degree=degree[j], kernel=kernel[k], decision_function_shape=decision_function_shape[l])
                # Fit SVM to the training data
                clf.fit(X_training, y_training)
                # make the SVM prediction for each test sample and compute its accuracy
                y_pred = clf.predict(X_test)
                correct_preds = 0
                for p in range(len(y_pred)):
                    if y_pred[p] == y_test[p]:
                        correct_preds = correct_preds + 1
                current_accuracy = correct_preds / len(y_pred)
                # check if the calculated accuracy is higher than the previously one calculated. 
                # If so, update the highest accuracy and print it together with the SVM hyperparameters
                if current_accuracy > highest_accuracy:
                    print(f"Highest SVM accuracy so far: {current_accuracy}, Parameters: c={c[i]}, degree={degree[j]}, kernel={kernel[k]}, decision_function_shape={decision_function_shape[l]}")
                    highest_accuracy = current_accuracy
