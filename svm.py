#-------------------------------------------------------------------------
# AUTHOR: Andrew Sanford
# FILENAME: svm.py
# SPECIFICATION: Train SVC model
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 Minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from ast import Str
from sklearn import svm
import numpy as np
import pandas as pd

highest_accuracy = 0

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

for cval in c:
    for degreeval in  degree:
        for kernelval in kernel:
           for decisionval in decision_function_shape:
                test_total=0
                test_right=0
                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                #--> add your Python code here
                svm_model = svm.SVC(C=cval, degree=degreeval,kernel=kernelval,decision_function_shape=decisionval)

                #Fit SVM to the training data
                #--> add your Python code here
                svm_model.fit(X=X_training,y=y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                #--> add your Python code here
                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    test_total += 1
                    if svm_model.predict([x_testSample]) == y_testSample:
                        test_right += 1

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                if test_right/test_total > highest_accuracy:
                    highest_accuracy = test_right/test_total

                    print(f"Highest SVM accuracy so far: {highest_accuracy}, Parameters: c={cval}, degree={degreeval}, kernel= {kernelval}, decision_function_shape = {decisionval}")




