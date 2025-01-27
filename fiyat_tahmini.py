import pandas as pd
import numpy as np
import os


df = pd.read_csv("hepsiemlak/01.csv")


def drop_colums(df, cols):
    for col in cols:
        try:
            df.drop([col], axis=1, inplace=True)
        except Exception as e:
            print(f"Error: {e}")
            
cols = ['img-link href', 'list-view-image src', 'photo-count', 'list-view-title', 'left', 'img-wrp href',
       'he-lazy-image src', 'wp-btn', 'listing-card--owner-info__firm-name',
       'he-lazy-image src 3']

drop_colums(df, cols)

df['city'] = df["list-view-location"].str.split("/").str[0]
df['district'] = df["list-view-location"].str.split("/").str[1]
df['neighbourhood'] = df["list-view-location"].str.split("/").str[2]

drop_colums(df, ['list-view-location'])

print(df.info())















