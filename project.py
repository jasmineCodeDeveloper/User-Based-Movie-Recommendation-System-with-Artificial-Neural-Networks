# -*- coding: utf-8 -*-
"""
Created on Fri May  8 01:56:52 2020

@author: emre9
"""


import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, KFold
sns.set_style("ticks")

#df1 = pd.read_csv("netflix-prize-data/movie_titles.csv")

df = pd.read_csv("combined_data_1.csv", header = None, names = ['Cust_Id','Rating'], usecols = [0,1], nrows = 280167)
#date kullanadık
df['Rating'] = df['Rating'].astype(float)


print('Dataset 1 shape: {}'.format(df.shape))
print('-Dataset examples-')
print(df.iloc[::100000, :])

# %% Data Viewing
p = df.groupby('Rating')['Rating'].agg(['count']) # Her bir raiting değerine oy veren kullanıcı sayısı

# get movie count
movie_count = df.isnull().sum()[1] # 1. index yani filmlerin sayısı

# get customer count
cust_count = df['Cust_Id'].nunique() - movie_count # Toplam oy veren kullanıcı sayısı

# get rating count
rating_count = df['Cust_Id'].count() - movie_count # Toplam oy sayısı

ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
plt.axis('off')

for i in range(1,6):
    ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')
    
    
# %% Data Cleaning    
df_nan = pd.DataFrame(pd.isnull(df.Rating)) # Oy gözükmeyen satırlar true oylar false
df_nan = df_nan[df_nan['Rating'] == True] # True olanları listeliyor bu sayede
df_nan = df_nan.reset_index() # Filmlere ait oyların başlangıç ve bitiş indexlerini görebiliyoruz

movie_np = []
movie_id = 1
# Hangi satırın hangi filme ait oy olduğunu yazan bir movie_np dizisi oluşturuyor
# Bu sayede örneğin 134589. indexin 23 id li filme ait oy olduğunu biliyoruz
for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
    # numpy approach
    temp = np.full((1,i-j-1), movie_id)
    movie_np = np.append(movie_np, temp)
    movie_id += 1

# Account for last record and corresponding length
# numpy approach
# son filmi burada ekliyor(sebebini çözemedik)
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
movie_np = np.append(movie_np, last_record)

print('Movie numpy: {}'.format(movie_np))
print('Length: {}'.format(len(movie_np)))

# remove those Movie ID rows
# 1:,2: gibi satırları silip Raiting sütunun yanına o satıra ait movie_id değerlerini ekliyor
# bu işlemden sonra df matrisi tamamen oy sayısı kadar satıra sahip oluyor
df = df[pd.notnull(df['Rating'])]

df['Movie_Id'] = movie_np.astype(int)
df['Cust_Id'] = df['Cust_Id'].astype(int)
print('-Dataset examples-')
print(df.iloc[::100000, :])

# %% Data Slicing

f = ['count','mean']
# movie_benchmark bir filme oy veren en az ortalamayı hesaplıyor 
# drop_movie_list bu ortalamanın altında kalanlar
df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.8),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

print('Movie minimum times of review: {}'.format(movie_benchmark))
# cust_benchmark bir kullanıcının verdiği oylardan en az ortalama gereksinimi hesaplıyor
# drop_cust_list bu ortalamanın altında kalanlar
df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.8),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

# Manual Testing Variables
Manual_df = df

print('Customer minimum times of review: {}'.format(cust_benchmark))
# yukarıda hesaplananları listeden çıkarıyor
print('Original Shape: {}'.format(df.shape))
df = df[~df['Movie_Id'].isin(drop_movie_list)]
df = df[~df['Cust_Id'].isin(drop_cust_list)]
print('After Trim Shape: {}'.format(df.shape))
print('-Data Examples-')
print(df.iloc[::100000, :])


df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')

print(df_p.shape)


# %% Data Mapping
# =============================================================================
# df_title = pd.read_csv("netflix-prize-data/movie_titles.csv", encoding = "ISO-8859-1", header = None, names = ['Movie_Id', 'Year', 'Name'], nrows = 60)
# df_title.set_index('Movie_Id', inplace = True)
# print (df_title.head(10))
# 
# 
# reader = Reader()
# 
# # get just top 100K rows for faster run time
# # =============================================================================
# # data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:50000], reader)
# # kf = KFold(n_splits=3)
# # kf.split(data)
# # 
# # svd = SVD()
# # cross_validate(svd, data, measures=['RMSE', 'MAE'])
# # =============================================================================
# # =============================================================================
# # 
# # 1:
# #     111222, 5
# #     222111, 3
# #     
# # 2:
# #     111222, 3
# #     222111, 4
# #     
# # 
# #     
# # 111222[5,3]
# # 222111[3,4]
# # =============================================================================
# df_785314 = df[(df['Cust_Id'] == 588844) & (df['Rating'] == 5)]
# df_785314_2 = df[(df['Cust_Id'] == 588844) & (df['Rating'] == 5)]
# df_785314 = df_785314.set_index('Movie_Id')
# df_785314 = df_785314.join(df_title)['Name']
# print(df_785314)
# 
# user_785314 = df_title.copy()
# user_785314 = user_785314.reset_index()
# user_785314 = user_785314[~user_785314['Movie_Id'].isin(drop_movie_list)]
# =============================================================================

# getting full dataset
# =============================================================================
# data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']], reader)
# 
# trainset = data.build_full_trainset()
# svd.fit(trainset)
# 
# #plot data
# # =============================================================================
# # plt.plot(df_785314_2.Cust_Id, df_785314_2.Rating,color = "black",label = "poly")
# # plt.legend()
# # plt.scatter(df_785314_2.Cust_Id, df_785314_2.Rating)
# # plt.xlabel("Cust_Id")
# # plt.ylabel("Rating")
# # plt.show()
# # =============================================================================
# 
# user_785314['Estimate_Score'] = user_785314['Movie_Id'].apply(lambda x: svd.predict(588844, x).est)
# 
# user_785314 = user_785314.drop('Movie_Id', axis = 1)
# 
# user_785314 = user_785314.sort_values('Estimate_Score', ascending=False)
# print(user_785314.head(10))
# 
# =============================================================================

# %% Manual YSA Algorithm


# Manual Testing Variables
U = df_cust_summary[df_cust_summary['count'] > 0.0].index
N = len(U)
I = df_movie_summary[df_movie_summary['mean'] > 0.0].index
M = len(I)

Manual_df_p = pd.pivot_table(Manual_df,values='Rating',index='Cust_Id',columns='Movie_Id')
