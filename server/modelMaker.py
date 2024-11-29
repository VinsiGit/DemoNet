import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Directory to save models
MODEL_SAVE_DIR = './models'
TARGET_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 20

# Ensure the directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def create_improved_cnn():
    model = models.Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)))  # Fixed size input
    
    # Convolutional Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    

    # Global Average Pooling to reduce variable size feature maps to a fixed vector
    model.add(layers.GlobalAveragePooling2D())
    
    # Fully Connected Layers
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    
    # Output Layer: Produces a single integer
    model.add(layers.Dense(1, activation='linear'))  # Use sigmoid or relu as needed
    
    return model

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = []
    years = []
    areas = []
    
    for item in data:
        image_path = item['image_path']
        year = item['year']
        area = item['area']

        image = load_img('./data' + image_path, target_size=TARGET_SIZE)  # Resize to a fixed size
        image = img_to_array(image)
        images.append(image)
        
        years.append(year)
        areas.append(area)
    
    images = np.array(images)
    years = np.array(years)
    areas = np.array(areas)
    
    return images, years, areas

def train_model(train_data, train_labels, model_name):
    model = create_improved_cnn()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    model.fit(train_data, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)

    model_save_path = os.path.join(MODEL_SAVE_DIR, model_name)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

def main():
    images, years, areas = load_data('./data/data.json')
    train_model(images, years, 'model_year.h5')
    train_model(images, areas, 'model_area.h5')

if __name__ == "__main__":
    main()