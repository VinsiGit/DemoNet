import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# Directory to save models and scalers
MODEL_SAVE_DIR = './models'
SCALER_SAVE_DIR = './scalers'
TARGET_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 20  # Increase the number of epochs

# Ensure the directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(SCALER_SAVE_DIR, exist_ok=True) 

def create_improved_cnn():
    model = models.Sequential()
    
    # Input Layer
    model.add(layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)))  # Fixed size input
    
    # Convolutional Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())
    
    # Global Average Pooling to reduce variable size feature maps to a fixed vector
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten()),

    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    
    # Output Layer: Produces a single integer
    model.add(layers.Dense(1, activation='linear'))  # Use sigmoid or relu as needed
    
    return model

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_paths = []
    years = []
    areas = []

    for item in data:
        image_path = item['image_path']
        year = item['year']
        area = item['area']

        image_paths.append(image_path)
        years.append(year)
        areas.append(area)

    return image_paths, years, areas

def preprocess_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, TARGET_SIZE)
    image = image / 255.0  # Normalize to [0, 1]
    return image, label

def prepare_dataset(df, base_path, label_column, scaler, is_training=True):
    file_paths = df["filename"].apply(lambda x: os.path.join(base_path, x[1:]))  # Remove leading slash
    labels = scaler.transform(df[[label_column]])
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def train_model(train_dataset, val_dataset, model_name, scaler):
    model = create_improved_cnn()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
    model.summary()
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_name)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

def main():
    # Load data from JSON file
    image_paths, years, areas = load_data('./data/data.json')

    # Create DataFrame from loaded data
    df = pd.DataFrame({
        'filename': image_paths,
        'year': years,
        'area': areas
    })

    # Normalize the labels
    year_scaler = MinMaxScaler()
    area_scaler = MinMaxScaler()
    df['year'] = year_scaler.fit_transform(df[['year']])
    df['area'] = area_scaler.fit_transform(df[['area']])

    # Save the scalers
    joblib.dump(year_scaler, os.path.join(SCALER_SAVE_DIR, 'year_scaler.pkl'))
    joblib.dump(area_scaler, os.path.join(SCALER_SAVE_DIR, 'area_scaler.pkl'))

    # Split DataFrame into training and validation sets
    train_df = df.sample(frac=0.7, random_state=42)
    val_df = df.drop(train_df.index)

    # Prepare datasets for 'year' and 'area'
    train_dataset_year = prepare_dataset(train_df, base_path="./data", label_column="year", scaler=year_scaler, is_training=True)
    val_dataset_year = prepare_dataset(val_df, base_path="./data", label_column="year", scaler=year_scaler, is_training=False)

    train_dataset_area = prepare_dataset(train_df, base_path="./data", label_column="area", scaler=area_scaler, is_training=True)
    val_dataset_area = prepare_dataset(val_df, base_path="./data", label_column="area", scaler=area_scaler, is_training=False)

    # Train models
    
    train_model(train_dataset_year, val_dataset_year, 'model_year.h5', year_scaler)
    train_model(train_dataset_area, val_dataset_area, 'model_area.h5', area_scaler)


if __name__ == "__main__":
    main()