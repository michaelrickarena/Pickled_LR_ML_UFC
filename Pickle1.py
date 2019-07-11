from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import warnings
import joblib
import pickle
import csv


warnings.simplefilter(action='ignore', category=FutureWarning)


clf_load=joblib.load('Real1.pkl')

new_data = pd.read_csv('C:/Users/micha/Desktop/Linear_Regression_13.csv') #untrained data

cleaned_data = new_data.drop(['Fighter','WeightClass', 'My Proj','Consistency','Value'], axis=1)

# print(cleaned_data.dtypes)

x= new_data[['Fight #','Odds', 'Win %','Cost','Avg Score', 'ITD Odds']]
y= new_data['Actual Score']

print(clf_load.score(x,y))
clf_load.predict(x)

Excel=clf_load.predict(x)


for i in Excel:
	print(i)
