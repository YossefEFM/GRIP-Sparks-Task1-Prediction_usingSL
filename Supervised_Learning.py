# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 02:07:02 2022

@author: Yossef
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import *
from sklearn.linear_model import *

import matplotlib.pyplot as plt
import seaborn as sns

#loading Data
dataset = pd.read_csv("student_scores - student_scores.csv")

features = dataset.iloc[:, :-1].values
label = dataset.iloc[:, 1].values

#Data Visualization
plt.scatter(features,label)
plt.title("Data set Visualzation")
plt.show()

#splitting data
x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=1)

#training model
regVal = LinearRegression().fit(x_train.reshape(-1, 1), y_train)

#Testing model in 9.25
prediction = regVal.predict((np.array([9.25])).reshape(-1, 1))

#printing Accuracy of the test
print('Linear Regression Accuracy:', prediction[0], '%')

#ploting the model after training
sns.regplot(features, label,color='red')
plt.title("Hours && percentage")
plt.show()

