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
       'he-lazy-image src', 'wp-btn', 'list-view-date', 'listing-card--owner-info__firm-name',
       'he-lazy-image src 3']

drop_colums(df, cols)

df['city'] = df["list-view-location"].str.split("/").str[0]
df['district'] = df["list-view-location"].str.split("/").str[1]
df['neighbourhood'] = df["list-view-location"].str.split("/").str[2]

drop_colums(df, ['list-view-location'])

df['celly'] = df['celly'].apply(lambda x: x.replace('Stüdyo', '1 + 0'))
df['celly'] = df['celly'].apply(lambda x: x.replace('\n', ''))
df['room'] = df['celly'].apply(lambda x: x.split('+')[0]).astype(int)
df['living_room'] = df['celly'].apply(lambda x: x.split('+')[1]).astype(int)

drop_colums(df, ['celly'])

df['celly 2'] = df['celly 2'].apply(lambda x: x.replace('.',''))
df['size'] = df['celly 2'].apply(lambda x: x.split(' ')[0]).astype(int)
drop_colums(df, ['celly 2'])


df['celly 3'] = df['celly 3'].apply(lambda x: x.replace('\n',' '))
df['celly 3'] = df['celly 3'].apply(lambda x: x.replace('Sıfır','0'))
df['age'] = df['celly 3'].apply(lambda x: x.split(' ')[0]).astype(int)
drop_colums(df, ['celly 3'])


replace_dic = {
    'Yüksek Giriş': '1. Kat',
    'Ara Kat': '3. Kat',
    'Yarı Bodrum': '-1. Kat',
    'En Üst Kat': '6. Kat',
    'Bodrum': '-1. Kat',
    'Giriş Katı': '1. Kat',
    'Zemin': '0. Kat',
    'Kot 1': '-1. Kat',
    'Villa Katı': '1. Kat',
    'Bahçe Katı': '0. Kat',
    'Kot 2': '-2. Kat',
    'Çatı Katı': '6. Kat',
    'Kot 3': '-3. Kat',
    None : '0. kat'
    
}

df['celly 4'] = df['celly 4'].replace(replace_dic.keys(), replace_dic.values())
df['floor'] = df['celly 4'].apply(lambda x: x.split('.')[0]).astype(int)
drop_colums(df, ['celly 4'])

df['price'] = df['list-view-price'].apply(lambda x: x.replace('.','')).astype(int)


print(df['price'].unique())















