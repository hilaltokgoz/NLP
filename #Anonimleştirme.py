#Anonimleştirme
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Sağlık veri kümesi
data = {
    'Name': ['John Doe', 'Jane Smith', 'Alice Johnson', 'Bob Brown'],
    'Age': [29, 34, 42, 31],
    'Disease': ['Diabetes', 'Hypertension', 'Asthma', 'Diabetes'],
    'City': ['New York', 'Los Angeles', 'New York', 'Chicago']
}

df = pd.DataFrame(data)
print("Orijinal Veri:")
print(df)

# 'Name' sütununu kaldırarak anonimleştirme
df = df.drop(columns=['Name'])
print("\nKimlik Bilgileri Kaldırıldı:")
print(df)

# Yaşları aralıklara ayırma (genelleştirme)
df['Age'] = pd.cut(df['Age'], bins=[20, 30, 40, 50], labels=['20-30', '30-40', '40-50'])
print("\nYaş Genelleştirildi:")
print(df)

# 'City' sütununa göre k-anonimlik sağlama
k = 2  # Her grup en az k kişi içermelidir #newyork 2 kez geçiyor yazar diğerleri other
df['City'] = df['City'].apply(lambda x: 'Other' if (df['City'] == x).sum() < k else x)

print("\nK-Anonimlik Uygulandı (k=2):")
print(df)


# Age sütununa rastgele gürültü ekleme
np.random.seed(42)
df['Age'] = df['Age'].apply(lambda x: str(int(x.split('-')[0]) + np.random.randint(-2, 2)) + "-40")
print("\nDiferansiyel Gizlilik ile Gürültü Eklendi:")
print(df)
