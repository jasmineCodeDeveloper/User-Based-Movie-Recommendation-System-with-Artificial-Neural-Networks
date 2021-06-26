import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Preprocess:
    data = None

    def __init__(self):
        sns.set_style("ticks")

        # df1 = pd.read_csv("netflix-prize-data/movie_titles.csv")
        df = pd.read_csv("combined_data_1.csv", header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1], nrows=280167)
        #df = pd.read_csv("netflix-prize-data/combined_data_1.csv", header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1], nrows=280167)
        df['Rating'] = df['Rating'].astype(float)

        # %% Data Viewing
        p = df.groupby('Rating')['Rating'].agg(['count'])  # Her bir raiting değerine oy veren kullanıcı sayısı

        # get movie count
        movie_count = df.isnull().sum()[1]  # 1. index yani filmlerin sayısı

        # get customer count
        cust_count = df['Cust_Id'].nunique() - movie_count  # Toplam oy veren kullanıcı sayısı

        # get rating count
        rating_count = df['Cust_Id'].count() - movie_count  # Toplam oy sayısı

       
        # %% Data Cleaning
        df_nan = pd.DataFrame(pd.isnull(df.Rating))  # Oy gözükmeyen satırlar true oylar false
        df_nan = df_nan[df_nan['Rating'] == True]  # True olanları listeliyor bu sayede
        df_nan = df_nan.reset_index()  # Filmlere ait oyların başlangıç ve bitiş indexlerini görebiliyoruz

        movie_np = []
        movie_id = 1
        # Hangi satırın hangi filme ait oy olduğunu yazan bir movie_np dizisi oluşturuyor
        # Bu sayede örneğin 134589. indexin 23 id li filme ait oy olduğunu biliyoruz
        for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
            # numpy approach
            #ilk 548 filme 1 diğerleri ne 2
            temp = np.full((1, i - j - 1), movie_id)
            movie_np = np.append(movie_np, temp)
            movie_id += 1
        # Account for last record and corresponding length
        # numpy approach
        # son filmi burada ekliyor(sebebini çözemedik)
        last_record = np.full((1, len(df) - df_nan.iloc[-1, 0] - 1), movie_id)
        movie_np = np.append(movie_np, last_record)

        # print('Movie numpy: {}'.format(movie_np))
        # print('Length: {}'.format(len(movie_np)))

        # remove those Movie ID rows
        # 1:,2: gibi satırları silip Raiting sütunun yanına o satıra ait movie_id değerlerini ekliyor
        # bu işlemden sonra df matrisi tamamen oy sayısı kadar satıra sahip oluyor
        df = df[pd.notnull(df['Rating'])]

        df['Movie_Id'] = movie_np.astype(int)
        df['Cust_Id'] = df['Cust_Id'].astype(int)

        # %% Data Slicing

        f = ['count', 'mean']
        # movie_benchmark bir filme oy veren en az ortalamayı hesaplıyor
        # drop_movie_list bu ortalamanın altında kalanlar
        df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
        df_movie_summary.index = df_movie_summary.index.map(int)
        movie_benchmark = round(df_movie_summary['count'].quantile(0.8), 0)
        drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

        # cust_benchmark bir kullanıcının verdiği oylardan en az ortalama gereksinimi hesaplıyor
        # drop_cust_list bu ortalamanın altında kalanlar
        df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
        df_cust_summary.index = df_cust_summary.index.map(int)
        cust_benchmark = round(df_cust_summary['count'].quantile(0.8), 0)
        drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

        df_all = df
        df = df[~df['Movie_Id'].isin(drop_movie_list)]
        df = df[~df['Cust_Id'].isin(drop_cust_list)]
# =============================================================================
#         df.loc[df.Weight == "155", "Name"] = "John"
# =============================================================================        
        df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')
      
        
        for col in df_p:
           df_p.loc[df_p[col] <= 2.0, col] = 0
           df_p.loc[df_p[col] > 2.0, col] = 1
        
        df_p = df_p.replace(np.nan,0)
       
        print(df_p.shape)
        
        # todo df array'i formatlanacak
        movie = df_p.iloc[:,5]
        
        self.select_movie  = movie.head(48000)
        self.predict_movie = movie.tail(12000)
        
        df_p = df_p.drop(df_p.columns[[5]], axis=1) 
        
        self.data_train = df_p.head(48000)
        self.data_predict = df_p.tail(12000)
        
        
        

    def get_training_inputs(self):
        return self.data_train # todo verinin yüzde 80'i return edilecek

    def get_training_outputs(self):
        return self.select_movie # todo oy array'inin yüzde 80'i return edilecek

    def get_test_inputs(self):
        return self.data_predict # todo verinin kalan kısmı

    def get_test_outputs(self):
        return self.predict_movie # todo oy array'inin kalan kısmı

