import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from PIL import Image
import os

# Birleştirilmiş CSV dosyasını okuma
combined_csv_path = 'tüm/combined_data.csv'
df = pd.read_csv(combined_csv_path)
df['filename_with_extension'] = df['filename'].apply(lambda x: x + '.jpg')

# Resim yollarını ve etiketleri alma
all_image_paths = df['filename_with_extension'].values
all_labels = df['class'].values

# Etiketleri sayısal değerlere dönüştürme
label_mapping = {label: idx for idx, label in enumerate(np.unique(all_labels))}
all_labels_numeric = np.array([label_mapping[label] for label in all_labels])

# Haritalama sözlüğünü yazdırma
print("Label Mapping:")
for label, idx in label_mapping.items():
    print(f"{label}: {idx}")

# Resimleri yükleme ve numpy dizisine dönüştürme
def load_and_preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')  # RGB formatına dönüştürme
        image = image.resize((150, 150))
        image = np.array(image)
        image = image / 255.0
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Klasördeki resim dosyalarının yolları
directory = 'dataset'
images = [load_and_preprocess_image(os.path.join(directory, path)) for path in all_image_paths]

# None değerlerini filtreleme
filtered_images_labels = [(img, label) for img, label in zip(images, all_labels_numeric) if img is not None]
images, all_labels_numeric = zip(*filtered_images_labels)

# Etiketleri one-hot encode etme
num_classes = len(label_mapping)
labels = tf.keras.utils.to_categorical(all_labels_numeric, num_classes=num_classes)

# Verileri numpy dizisine dönüştürme
images = np.array(images)

# Verileri eğitim ve doğrulama setlerine ayırma
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=directory,
    x_col="filename_with_extension",
    y_col="class",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    directory=directory,
    x_col="filename_with_extension",
    y_col="class",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Modeli oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(train_generator, epochs=25, validation_data=validation_generator)

# Modeli kaydetme
model.save('uçaksınıflandırmacsv.h5')

# Haritalama sözlüğünü kaydetme
with open('label_mapping.txt', 'w') as f:
    for label, idx in label_mapping.items():
        f.write(f"{label}: {idx}\n")
