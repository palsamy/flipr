import pandas as pd  
import numpy as np
import csv  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


with open('output_file01.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["People_ID","Infect_Prob"])

train_data = pd.read_csv('train_task1.csv')
x = train_data.iloc[0:,1:14].values
y = train_data.iloc[0:,14:].values

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x)
x= imputer.transform(x)

test_data = pd.read_csv('test_task1.csv')
X = test_data.iloc[0:,1:14].values
People_ID = test_data.iloc[0:,0:1].values

imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X)
X= imputer.transform(X)

regressor = LinearRegression()  
regressor.fit(x,y)

y_pred = regressor.predict(X)

with open('output_file01.csv','a',newline='') as file:
    for i in range(len(y_pred)):
        writer = csv.writer(file)
        writer.writerow([People_ID[i][0],y_pred[i][0]])

file.close()