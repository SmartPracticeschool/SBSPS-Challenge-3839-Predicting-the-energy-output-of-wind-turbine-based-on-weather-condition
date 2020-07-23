
import numpy as np
import pandas as pd
import pickle

dataset=pd.read_csv('C:/Users/Pradeepa/Desktop/csv/T1.csv'')

dataset['Wind Speed (m/s)'].fillna(0,inplace=True)

dataset['Theoretical_Power_Curve (KWh)'].fillna(dataset['Theoretical_Power_Curve (KWh)'].mean(),inplace=True)

dataset['Wind Direction (°)'].fillna(0,inplace=True)

X=dataset.iloc[:,2:]
y=dataset.iloc[:,1]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)

pickle.dump(regressor,open('model.pkl','wb'))
