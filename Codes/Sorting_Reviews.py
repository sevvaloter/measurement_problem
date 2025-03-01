import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def score_up_down_diff(up, down):
    return up - down

# Review 1 Score:
score_up_down_diff(600, 400)#200

# Review 2 Score
score_up_down_diff(5500, 4500) #1000

#Fonksiyonlaştırarak threshold belirleme
#Average Rating Değeri(UP RATIO)
# Score = Average rating = (up ratings) / (all ratings)

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400) #0.6
score_average_rating(5500, 4500) #0.55

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0) #1.0, 2 YORUM 2 UPP 1 DOWN
score_average_rating(100, 1) #0.9900 1 DOWN
#Terste bir iş var frekans yüksekliğini değerlendiremedi.

# 600-400 (up-down)
# 0.6 up oranı
# 0.5 0.7 yanılma payı(0.6 için güven aralığı,
#yüzde beş yanılma payıyla up oranı 0.5-0.7 arasında)
# 0.5 i skor olarak belirlerim.
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


wilson_lower_bound(600, 400) #0.5693
wilson_lower_bound(5500, 4500)#0.5402 ,fazlalık faydalılık anlamına gelmiyor demek ki,binary_rating inceliyoruz.

wilson_lower_bound(2, 0) #0.342
wilson_lower_bound(100, 1) #0.94603

#yorumların faydalı veya faydasız bulunmasını df e çevirdik.
up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})



# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],
                                                                             x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)



comments.sort_values("score_pos_neg_diff", ascending=False)
comments.sort_values("score_average_rating", ascending=False)
comments.sort_values("wilson_lower_bound", ascending=False)