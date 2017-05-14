# -*- coding: utf-8 -*-

"""
@author - Radhakrishna Jella

This program is for Regression and Gradient descent computation.

Regression model
   create hypothesis - create an equation for output variable
   Assume theta with initial values, theta = column matrix (2 by 1)
   Actaul input from training data is column matrix = (m by 1) m is no.of training sets.
   Target values from training data is column matrix = (m by 1) m is no.of training sets.
   Including bias term consider x_matrix as m by 2 - first column is bias coeff(1's)
                                                     second column is input data 
   Theta is 2 by1, x is m by 2 -> to do matrix multiplication
   Training rate = alpha - is a scaler
   Gradient is 2 by 1, which used to compute new thetas
   Maximum iterations are for limiting computation if error tolerance is not met
   
For each iteration, 
   hypothesis = theta0 + theta * x_feature1 = x_matrix * theta = m by2 * 2 by 1 = m by 1
   compare hypothesis ouput with target value to compute error
   error = hypothesis output - Target values (both matrix are of m by 1)
   cost = 1/2m* sum(error ** 2)
   
   gradient = 1/m * x_matrix.transpose * error = 2 by m * m by 1 = 2 by 1 
   
   update theta = theta - alpha * gradient
   
use new theta for computing hypothesis for next iteration, continue iterations
   until maximum iterations are completed or loss due to model is with in tolerance
"""

import numpy as np
import random

""" to generate input data with intercept and slope, with varaiance"""
def prepare_input_data(intercept, slope, variance, no_of_inputs):
    x = np.zeros(shape = (no_of_inputs,2)) # x is m by 2. 
    y = np.zeros(shape = (no_of_inputs,1))
    
    for index in range(no_of_inputs):
        x[index][0] = 1     # 0th column represents 1 for bias terms for each row
        x[index][1] = index # 1st column represents index for sample for each row 
        y[index] = (intercept + index * slope) + random.uniform(0, 1) * variance
    
    return x, y
    
"""
   hypothesis = theta0 + theta * x_feature1 = x_matrix * theta = m by2 * 2 by 1 = m by 1
   compare hypothesis ouput with target value to compute error
   error = hypothesis output - Target values (both matrix are of m by 1)
   cost = 1/2m* sum(error ** 2)
   
   gradient = 1/m * x_matrix.transpose * error = 2 by m * m by 1 = 2 by 1 
   
   update theta = theta - alpha * gradient
"""
def gradient_descent(x, y, theta, alpha, no_of_inputs):
    hypothesis = x.dot(theta)
    error = hypothesis - y
    cost = 1/(2 * no_of_inputs)*np.sum(error ** 2)
    gradient = 1/no_of_inputs * x.T.dot(error)
    theta = theta - alpha * gradient
    return theta, cost


if __name__ == "__main__":
    #preparing training data
    """ By updating following members this program can be worked for 
        different sizes of data sets with different aplha, iteration counts"""
    intercept = 20
    slope = 32
    variance = 2
    no_of_inputs = 2
    alpha = 0.1
    no_of_iterations = 200
    cost = 0
    
    #generates data set which fits close to y=intercept+slope*x
    x, y = prepare_input_data(intercept, slope, variance, no_of_inputs)
    theta = np.ones(shape = (2,1))
    
    #fit model 
    for i in range(no_of_iterations) :
        theta, cost = gradient_descent(x, y, theta, alpha, no_of_inputs)
        if i%10 == 0:
            print("theta", theta)
            print ("cost", cost)
            print("iteration", i)
            input("enter some input")
        if (cost < 1e-6):
            print(" theta has been found")
            break
    
    print("theta", theta)
    print("cost", cost)




            
        
    
    
    
    
    
    
    
 
    
    

