import pandas as pd 
import numpy as np


df = pd.read_csv("hepsiemlak/02.csv")


df['celly 3'] = df['celly 3'].astype(str)

df['celly 3'] = df['celly 3'].replace('nan', '0')

df['celly 3'] = df['celly 3'].apply(lambda x: x.replace('\n',' '))
df['celly 3'] = df['celly 3'].apply(lambda x: x.replace('Sıfır','0'))
df['celly 3'] = df['celly 3'].apply(lambda x: x.replace('Sıfır Bina','0'))
df['age'] = df['celly 3'].apply(lambda x: x.split(' ')[0]).astype(int)



