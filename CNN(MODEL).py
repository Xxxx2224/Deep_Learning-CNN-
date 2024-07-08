import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, AveragePooling2D, MaxPooling2D, BatchNormalization  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet121
tf.keras.backend.clear_session()
datagen = ImageDataGenerator(
    rescale=1./255,
    
    validation_split=0.2  # Verinin %20'sini doğrulama için ayır
)


train_generator = datagen.flow_from_directory(
    'yenidata',
    target_size=(500,500),
    batch_size=1,
    class_mode='categorical',
    subset='training',  
    shuffle=True  
)

validation_generator = datagen.flow_from_directory(
    'yenidata',
    target_size=(500,500),
    batch_size=1,
    class_mode='categorical',
    subset='validation',  
    shuffle=True  
)
conv_base = DenseNet121(
    weights='imagenet',
    include_top = False,
    input_shape=(500,500,3),
    pooling='avg'
)
conv_base.trainable = False

class_count = len(list(train_generator.class_indices.keys()))
input_shape = (500,500, 3)
num_classes = len(train_generator.class_indices)
model = Sequential()
model.add(conv_base)
model.add(BatchNormalization())
model.add(Dropout(0.35))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.35))
model.add(Dense(120, activation='relu'))
model.add(Dense(5, activation='softmax'))
class_count = len(train_generator.class_indices)
print("Sınıf sayısı:", class_count)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('CNNsave11111.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1000, min_lr=1e-6)
print("Sınıf indexleri:", train_generator.class_indices)


history = model.fit(
    train_generator,
    epochs=2500,  # Epoch sayısını artırdık
    validation_data=validation_generator,
    callbacks=[EarlyStopping(patience=0)]
)

# Eğitim ve doğrulama kayıplarını ve doğrulukları görselleştirme (isteğe bağlı)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Son Modeli Kaydetme
model.save('CNNsave11111.keras')

# Sınıf indexlerini yazdır
print("Sınıf indexleri:", train_generator.class_indices)
