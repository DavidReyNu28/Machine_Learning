#!/usr/bin/env python
# coding: utf-8

# Libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


########## Read the Dataset with pandas library 

data=pd.read_csv("Datasets/Electricity_Norm/electricity-normalized.csv")

# Shuffle data
# 
######### In order to get some random train and test data ###################

data=data.sample(len(data))
print(data)


# Get from the dataset just the labels that we need to implement the model

#Train data
X=["nswprice","nswdemand","vicprice","vicdemand"];Y="transfer"
xtrain=data[X][0:2000] 
ytrain=data[Y][0:2000]

#Test data
xtest=data[X][2000:2100] 
ytest=data[Y][2000:2100]


# Hypothesis function
def hyp(params, x):
    h = 0
    for i in range(len(params)):
        h+=(params[i]*x[i])
    return h


############## Error calc ##############3 
# It gives the actual improvement of the error mean and keep it for further plotting
evol_error = []
def  error(params, x, y):
    e_acum=0
    for i in range(len(x)):
        n=hyp(params, x[i])
        e_acum+=(n - y[i])**2 #mean square error
    error_mean= e_acum / len(x)
    evol_error.append(error_mean)
    print("E_meam = %f" %(error_mean))


############# Gradient Descent ##############
# This function updates the params to improve performance
def gd(params, x, y, a):
    tmp=list(params)
    for j in range(len(params)):
        acum=0;
        for i in range(len(x)):
            error = hyp(params, x[i]) - y[i]
            acum+= error * x[i][j]
        tmp[j] = params[j] - a * (1/len(x)) * acum
    return tmp # Return new parameters


################## Main ###################
    
#### Training the Model

evol_error = [] #Save the error evolution to plot it
epoch = 0 #inicialized epochs 
a = 0.0006 # Learning rate
limit = 3000 #3000 epochs to finish

params=[0, 0, 0, 0, 0]

xlist=xtrain.values.tolist()
ylist=ytrain.values.tolist()

for i in range(len(xlist)):
    xlist[i]=[1] + xlist[i]

while True:
    old = list(params)
    params=gd(params, xlist, ylist, a)
    error(params, xlist, ylist)
    epoch+=1
    if(old==params or epoch==limit):
        print ("Weights:")
        print (params)
        break

plt.plot(evol_error)


# Test
# 
# Evaluate with the test data, show graph relationa and error mean

x = np.linspace(0,len(ytest),len(ytest))
p=(hyp(params,[1, xtest["nswprice"], xtest["nswdemand"], xtest["vicprice"], xtest["vicdemand"]]))

testem_error=0
p=p.values.tolist()
y=ytest.values.tolist()
for i in range(len(x)):
    testem_error+= (p[i]-y[i])**2

print("Test Errormean = %f"%(testem_error/len(x)))


################# User queries ####################33
### User iputs in order to make a prediction

x1=float(input("Enter nswprice:"))
x2=float(input("Enter nswdemand:"))
x3=float(input("Enter vicprice:"))
x4=float(input("Enter vicdemand:"))
prediction = (hyp(params,[1,x1,x2,x3,x4]))
print(prediction)