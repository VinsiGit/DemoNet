import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Directory to save models
MODEL_SAVE_DIR = './models'

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


def train_model(train_data, train_labels, model_name):
    model = create_variable_size_cnn()
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(train_data, train_labels, epochs=10, batch_size=32)
    
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_name)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


def load_and_predict(model_name, data):
    model_path = os.path.join(MODEL_SAVE_DIR, model_name)
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(data)
    return predictions.astype(int)  # Ensure the output is an integer


# Example usage:
# train_data and train_labels should be your training dataset and labels
# train_model(train_data, train_labels, 'model_opp.h5')
# train_model(train_data, train_labels, 'model_year.h5')

# To load and predict
# predictions = load_and_predict('model_opp.h5', test_data)
# predictions = load_and_predict('model_year.h5', test_data)

def predict_year(image):
    predictions = load_and_predict('model_year.h5', image)
    return predictions

def predict_opp(image):
    predictions = load_and_predict('model_opp.h5', image)
    return predictions