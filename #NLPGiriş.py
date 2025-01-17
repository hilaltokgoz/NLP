#NLP Giriş
#textVectorization: veri ön işleme adımlarını kolayca yapar.
#metni küçük harfe çevirir.
#noktalama işaretini kaldırır.
#boşluklara göre metni parçalar.

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

text_vectorization = TextVectorization() #CPU'da çalışır
data = [
    'Bugün hava çok güzel',
    'Ali, efe ve CEM çay içecek',
    'Selam SÖYLE!'
]

text_vectorization.adapt(data) #verideki kelimelerden sözlük oluşturuldu.
#her token indexlendi

vocabulary = text_vectorization.get_vocabulary()
print('Sözlük:', vocabulary)
# Sözlük: ['', '[UNK]', 'çok', 'çay', 've', 'sÖyle', 'selam', 'içecek', 'hava', 'güzel', 'efe', 'cem', 'bugün', 'ali']
#UNK-> 1. index sözlükte olmayan kelimeler için ayrıldı.

vectorized_text = text_vectorization(data)#vector(sayısal) temsiline dönüştürme
print(vectorized_text)
# Tensor(
# [[12  8  2  9  0  0] ->1. cümlenin ilk indexi 12 sözlükten bakarsak bugün kelimesine denk  gelir.
#  [13 10  4 11  3  7]
#  [ 6  5  0  0  0  0]], shape=(3, 6), dtype=int64)

#model oluşturma
model = tf.keras.models.Sequential([
tf.keras.Input(shape = (1,), dtype =tf.string),  #Input Katmanı, veri boyutu:3, veri tipi: string
text_vectorization # TextVectorization katmanı
])

# Modeli test et
test_data = tf.constant(['Bugün hava ne güzel', 'Selam SÖYLE!', 'Ali ve Cem'])
output = model.predict(test_data)
print(output)
# [[12  8  1  9] ->1 o kelime('ne') sözlükte yok demek
#  [ 6  5  0  0]
#  [13  4 11  0]]