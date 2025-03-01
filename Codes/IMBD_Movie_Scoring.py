mport pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/content/dataset/movies_metadata.csv",low_memory=False)
df = df[["title","vote_average","vote_count"]]
df.head()
df.shape

df.sort_values("vote_average", ascending=False).head(20) #ortalamaya göre sıralama mantıklı değil

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T
#filmlerin ortalama oy sayısı 100 civarında

#yorum oy sayısı 400 den büyük olanlara göre sırala
df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)
#ortalamayı oy sayısına göre elle dikkate almak zor ve gereksiz dolayısıyla
#ortalamayı standartlaştıralım.

from sklearn.preprocessing import MinMaxScaler

df["vote_count_score"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

#vote average ve vote count score arasında da bağlantı kurmak istiyoruz.
df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values("average_count_score", ascending=False).head(20)

#############
#IMDB Ağırlıklı Derecelendirme ( IMDB Weighted Rating )
#############

M = 2500 #minimum gereken oy sayısı
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df.sort_values("average_count_score", ascending=False).head(10)

weighted_rating(7.40000, 11444.00000, M, C) #deadpool için inceleme weighted_rating için

weighted_rating(8.10000, 14075.00000, M, C) #inception

weighted_rating(8.50000, 8358.00000, M, C) #esaretin bedeli
""" Yapmış olduğumuz değere göre farklı değerler aldık wr ile"""

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)



##############
#Bayesian Average Rating Score( BAR Score )
##############

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

bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

#baba
bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]
#problemli satırlardan kurtulma


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)
