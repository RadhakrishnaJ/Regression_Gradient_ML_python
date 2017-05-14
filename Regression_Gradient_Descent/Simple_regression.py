# -*- coding: utf-8 -*-

""" @author - Radhakrishna Jella

Simple linear regression with one feature is explained here. 
By chnaging input file, adding new features it can be extended 
for multiple variables. 

Using pandas for file reading, so that feature set can be easily extended.
For Linear regression using SKLEARN. 

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics


training_data = pd.read_csv("Input_data.txt")

print(training_data.describe())
print(training_data.head())

model_input = training_data['x']  
"""use list of fetaure names for multiple features"""
model_target = training_data['y']


Regr_model = LinearRegression()
print("fit")
Regr_model.fit(model_input.reshape(-1,1), model_target)
""" Reshape is not needed is multiple features exists"""

print("Model Coefficients")
print("intercept", Regr_model.intercept_)
print("coefficient", Regr_model.coef_)

predicted_y = Regr_model.predict(model_input.reshape(-1,1))
print("metrics - r2")
print(metrics.r2_score(model_target, predicted_y))
print("Program completed successfully")
