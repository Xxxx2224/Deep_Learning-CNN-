import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('CNNsave11111.keras', monitor='val_loss', save_best_only=True)
new_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)


new_generator = new_datagen.flow_from_directory(
    'Data/crop',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='training',  
    shuffle=True  
) 
validation_generator = new_datagen.flow_from_directory(
    'Data/crop',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='validation',  
    shuffle=True 
)
# Modeli Yükleme
model = load_model('CNNsave11111.keras')
model.summary()


# Modeli Derleme
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
print("Sınıf indexleri:", new_generator.class_indices)

# Modeli Yeni Verilerle Eğitme
history = model.fit(
    new_generator,
    epochs=200,  
    validation_data=validation_generator,
    callbacks=[early_stopping, model_checkpoint]
)

# Geliştirilmiş Modeli Kaydetme
model.save('CNNsave11111.keras')

# Sınıf Indexlerini Yazdırma
print("Sınıf indexleri:", new_generator.class_indices)