
#Import Libraries
import pandas as pd
import math
import scipy.stats as st
import sklearn
from sklearn.preprocessing import MinMaxScaler

#Tabloyu Gözlemleme
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#Kurs datasını içeren verisetini gözlemleme
df = pd.read_csv("/content/sample_data/course_reviews.csv")
df.head()
#df.shape

# Rating Dağılımı

df["Rating"].value_counts()

df["Questions Asked"].value_counts()

#Sorulan soru kırılımında puan kırılımı
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                               "Rating": "mean"})

df.head()

#Time-Based Weighted Average (Puan Zamanlarına Göre Ağırlıklı Ortalama)
"""Ortalama Puan Ortalaması,ilgili ürünlerle ilgili son zamanlardaki trendi(memnuniyet trendini) kaçırabiliriz.
Güncel trendi ortalamaya daha güzel bir şekilde yansıtmalıyız"""

df["Rating"].mean()
df.head()
df.info()
#timestamp değişkeni object olarak geldiğinden değiştirmeliyiz
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
# Bugünün tarihini belirleyip bu tarihten yorumlara bak.
#str cinsinden bir değişkeni datetime değişkenine çevirdik
current_date = pd.to_datetime('2021-02-10 0:0:0')



df["days"] = (current_date - df["Timestamp"]).dt.days
df[df["days"] <= 30].count() #son 30 günde yapılan yorumlar

#Son 30 gündeki Rating değişkeninin ortalaması
df.loc[df["days"] <= 30, "Rating"].mean()

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()
df.loc[(df["days"] > 180), "Rating"].mean()

#son zamanlarda kursun memnuniyetinde artış var

"""aşağıdan kod yazmaya devam etmek için ters/
yüzdeliklerle dönüşmüş ana puana erişmek istiyoruz.
Vereceğimiz aralık sayısı arttıkça hareket edebileceğimiz aralık azalır 
Bu aralıkların 100 olması lazım.
En son zamanda yapılan yorumlar daha önemli olduğu için daha yüksek katsayı verdik
"""
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22/100



#Her seferinde ağırlıkları belirlemektense ağırlıklar için fonksiyon yazdık.

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100

time_based_weighted_average(df)
time_based_weighted_average(df, 30, 26, 22, 22)


#####
#Kullanıcı Temelli Ağırlıklı Ortalama ( User-Based Weighted Average )
#####
#Kursu Açıp kapatan insanla kursu bitiren öğrencinin etkisi aynı mı olmalı?
#Mesela IMBD 'de yüzlerce film izleyip yorum yapan insanla bir kere yorum yapan aynı olmamalı

df.head()

df.groupby("Progress").agg({"Rating": "mean"})
#ilerlemeye göre gruplama
#ilerleme durumuyla verilen puan arasında ilişki doğru orantılı


#elle ağırlık vererek yeni bir puanlama oluşturma
df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100
#Yorumlarda zaman tabanlı veya kullanıcı kalitesine göre ağırlıklandırmalar yapılır.

#Bizim buradaki metriğimiz izlenme oranı
def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["Progress"] > 75), "Rating"].mean() * w4 / 100

user_based_weighted_average(df, 20, 24, 26, 30)

#User quality scorlar yapmalıyız burada önem verdiğimiz ilerleme oldu.


##############
#Ağırlıklı Derecelendirme ( Weighted Rating )
##############
#zamandan ve userdan gelen değerlerin ağırlığını da dikkate aldık

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60) #user quality ye önemi arttıralım.

"""  
User quality ye şirketler metrikler belirlerken sitede geçirilen zaman,
yorum sayısı ,süreklilik vb. değişkenlere göre kullanıcı değerlerini belirliyorlar.
"""

df.loc[(df["Progress"] > 50) & (df["Progress"] <= 75), "Rating"].mean() * 0.35

#yanlış kod yanlışlığını sorgula
#df["Progress"].loc[(df["Progress"] > 50) & (df["Progress"] <= 75)].mean() * 0.35