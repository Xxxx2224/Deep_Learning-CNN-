import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Eğitilmiş Modeli Yükleme
model = load_model('CNNsave11111.keras')

model.summary()
# Yeni Veri İçin Data Generator
new_data_gen = ImageDataGenerator(rescale=1./255)
class_indices = {
    'A10': 0, 'A400M': 1, 'AG600': 2, 'AV8B': 3, 'B1': 4, 'B2': 5, 'B52': 6, 
    'Be200': 7, 'C130': 8, 'C17': 9, 'C2': 10, 'C5': 11, 'E2': 12, 'E7': 13, 
    'EF2000': 14, 'F117': 15, 'F14': 16, 'F15': 17, 'F16': 18, 'F18': 19, 'F22': 20, 
    'F35': 21, 'F4': 22, 'H6': 23, 'J10': 24, 'J20': 25, 'JAS39': 26, 'JF17': 27, 
    'KC135': 28, 'MQ9': 29, 'Mig31': 30, 'Mirage2000': 31, 'P3': 32, 'RQ4': 33, 
    'Rafale': 34, 'SR71': 35, 'Su24': 36, 'Su25': 37, 'Su34': 38, 'Su57': 39, 
    'Tornado': 40, 'Tu160': 41, 'Tu22M': 42, 'Tu95': 43, 'U2': 44, 'US2': 45, 
    'V22': 46, 'Vulcan': 47, 'XB70': 48, 'YF23': 49
}

# Yeni Veri Yükleyici
new_data_directory = 'uploads'  # Tahmin edilecek yeni verinin klasörü
new_data_generator = new_data_gen.flow_from_directory(
    new_data_directory,
    target_size=(224, 224),
    batch_size=8,
    class_mode=None,  # Tahmin işlemi için class_mode'u None olarak ayarlayın
    shuffle=False  # Tahmin sırasında sıralamanın karışmaması için shuffle=False
)

# Tahminleri Yapma
predictions = model.predict(new_data_generator)

# Tahmin edilen sınıfları alma
predicted_classes = np.argmax(predictions, axis=1)
index_to_class = {v: k for k, v in class_indices.items()}

# Sınıf İndekslerini Al
for index in predicted_classes:
    print(f'İndex:{index} Sınıf:{index_to_class[index]}')