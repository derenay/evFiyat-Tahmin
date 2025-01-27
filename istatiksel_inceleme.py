import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data.csv")


df['city'] = df['city'].astype('category')
df['district'] = df['district'].astype('category')
df['neighbourhood'] = df['neighbourhood'].astype('category')

colums =df.select_dtypes(include=[np.number]).columns

min_values = []
max_values = []

for column in colums:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    min_value = Q1 - 1.5 * IQR
    max_value = Q3 + 1.5 * IQR
    
    min_values.append(min_value)
    max_values.append(max_value)
    
    print(f"{column} min: {min_value}, max: {max_value}")
    


for i, column in enumerate(colums):
    df = df[(df[column] >= min_values[i]) & (df[column] <= max_values[i])]
    
    
df.to_csv("clear_data.csv")