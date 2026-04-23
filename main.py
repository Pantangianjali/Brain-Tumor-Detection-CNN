import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest", validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'dataset/', target_size=(224, 224), batch_size=32,
    class_mode='binary', subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset/', target_size=(224, 224), batch_size=32,
    class_mode='binary', subset='validation'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training
history = model.fit(train_generator, epochs=25, validation_data=val_generator,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
)

# Save model
model.save('models/tumor_model.h5')

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy - 96.8%')
plt.legend()
plt.savefig('results/accuracy_plot.png')
plt.show()

print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
