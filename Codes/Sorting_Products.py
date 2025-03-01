##############
# Sorting Products(Ürün Sıralama)
##############


###########
#Kurs Sıralama
############

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/product_sorting.csv")
print(df.shape)
df.head(10)

df.sort_values("rating", ascending=False).head(20)
#alakasız sonuçlar da gelebilir ,puanı yüksek olan kurslarda satın alma rating
#ve comment değerleri alakasız olabiliyor bu faktörleri de göz önüne almalıyız.

df.sort_values("purchase_count", ascending=False).head(20)
df.sort_values("commment_count", ascending=False).head(20)

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])

df.describe().T
#aynı değişkeni comment için de yapalım.
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])
#Hassaşlaştırma ekleme mesela rating bizim için daha önemliyse 👀
(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)

#Bu hesapladığım son değer skor(tek başına bir değer değil bağımlı/bağıl değişken)


def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

#Veri Bilimi keywordlü kursları getir.
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)


##############
#Bayes Ortalama Derecelendirme Puanı ( Bayesian Average Rating Score )
##############
# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Rating
#(5 Yıldızın Dağılımına Göre Ürün Sıralama)
#import math
#import scipy.stats as st
#n girilecek olan yıldızları skorları ve skorlara ait gözlenme frekanslarını ifade eder.
#confidence hesaplanacak olan z tablo değerine ilişkin bir değer elde etmek için tanımlandı.

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df.head()

#bar_score ,bar_average, bar_sorting_score,bar_average_rating adı da verebilir.
#x değişkenleri temsil ediyor. axis=1 sütunlarda gezinme
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis=1)
df.sort_values("weighted_sorting_score", ascending=False).head(20)

df.sort_values("bar_score", ascending=False).head(20)
"""
bar_score ve rating yakın değerler almış dikkat et .
bar score bize sadece ratinge odaklanarak bir dağılım oluşturdu.
ama social prooflar gözden kaçtı sadece puanlara göre sıralama için bar score
kullanılabilir ama çok değişken için bir skor hesaplamak istediğimizde 
yetersiz kaldı.bar_score sadece puanların dağılımına odaklandı."""

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False)
#course1 daha üstte çıktı bunun sebebi düşük puanlardaki frekans daha az olduğu  için yukarda geldi.


##############
#Hybrid Sorting: BAR Score + Diğer Faktorler
##############
def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                     "2_point",
                                                                     "3_point",
                                                                     "4_point",
                                                                     "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score*bar_w/100 + wss_score*wss_w/100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)
df.sort_values("hybrid_sorting_score", ascending=False).head(20)
#wss ortalama iken bar_score potansiyeli ifade eder.
#course 1 potansiyel vad ediyor bunu da bar_score ile gözlemleriz bu kursun ilk 10'a girmesinin sebebi
#course 1 yeni olmasına rağmen bar_score sayesinde pazarda şansının olmasını sağladık.
#yeterli social proof olmayan yeni değişkenleri de sıralama içine almamızı sağlar.

#alakasız kursları çıkarmak
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)
