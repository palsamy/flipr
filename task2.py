# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:30:38 2020

@author: Lovely Pals
"""

import pandas as pd  
import numpy as np
import csv  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('task2.csv')
x = train_data.iloc[0:,1:7].values
y = train_data.iloc[0:,7:].values

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x)
x= imputer.transform(x)

test_data = pd.read_csv('task2.csv')
X = test_data.iloc[0:,2:8].values

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X)
X= imputer.transform(X)
People_ID = test_data.iloc[0:,0:8].values

regressor = LinearRegression()  
regressor.fit(x,y)

y_pred = regressor.predict(X)
y_pred=y_pred.astype(int)

with open('date27.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["People_ID","d20","d21","d22","d23","d24","d25","d26","d27"])

file.close()

with open('date27.csv','a',newline='') as file:
    for i in range(len(y_pred)):
        writer = csv.writer(file)
        writer.writerow([People_ID[i][0],People_ID[i][1],People_ID[i][2],People_ID[i][3],People_ID[i][4],People_ID[i][5],People_ID[i][6],People_ID[i][7],y_pred[i][0]])

file.close()

with open('output_file02.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["People_ID","Infect_Prob"])

train_data = pd.read_csv('train_task2.csv')
x = train_data.iloc[0:,1:14].values
y = train_data.iloc[0:,14:].values

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x)
x= imputer.transform(x)

test_data = pd.read_csv('test_task2.csv')
X = test_data.iloc[0:,1:14].values
People_ID = test_data.iloc[0:,0:1].values

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X)
X= imputer.transform(X)

regressor = LinearRegression()  
regressor.fit(x,y)

y_pred = regressor.predict(X)

with open('output_file02.csv','a',newline='') as file:
    for i in range(len(y_pred)):
        writer = csv.writer(file)
        writer.writerow([People_ID[i][0],y_pred[i][0]])

file.close()