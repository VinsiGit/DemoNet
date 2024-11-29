import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Directory to save models
MODEL_SAVE_DIR = './models'
TARGET_SIZE = (256, 256)
# Ensure the directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def create_variable_size_cnn():
    model = models.Sequential()
    
    # Input Layer with variable size: None allows for dynamic dimensions
    model.add(layers.Input(shape=(None, None, 3)))  # 3 for RGB channels
    
    # Convolutional Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Global Average Pooling to reduce variable size feature maps to a fixed vector
    model.add(layers.GlobalAveragePooling2D())
    
    # Output Layer: Produces a single integer
    model.add(layers.Dense(1, activation='linear'))  # Use sigmoid or relu as needed
    
    return model

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = []
    years = []
    opps = []
    
    for item in data:
        image_path = item['image_path']
        year = item['jaar']
        opp = item['opp']

        image = load_img('./data'+image_path, target_size=TARGET_SIZE)  # Resize to a fixed size
        image = img_to_array(image)
        images.append(image)
        
        years.append(year)
        opps.append(opp)
    
    images = np.array(images)
    years = np.array(years)
    opps = np.array(opps)
    
    return images, years, opps

def train_model(train_data, train_labels, model_name):
    model = create_variable_size_cnn()
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(train_data, train_labels, epochs=10, batch_size=32)
    
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_name)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

def main():
    images, years, opps = load_data('./data/data.json')
    train_model(images, years, 'model_year.h5')
    train_model(images, opps, 'model_opp.h5')

if __name__ == "__main__":
    main()