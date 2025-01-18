
#Duygu Analizi Uygulama
#Bu veri seti, psikolojik açıdan kendini kötü hisseden kişilerin düşünceleri ve psikologlarının onlara verdikleri cevaplardan oluşmaktadır.
#cevaplar pozitif mi negatif mi?
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#pd.set_option('display.max_colwidth', None)
df = pd.read_csv("train.csv")
print(df.head(3))

text_vectorization = TextVectorization() ,
data = [df]

text_vectorization.adapt(data)
