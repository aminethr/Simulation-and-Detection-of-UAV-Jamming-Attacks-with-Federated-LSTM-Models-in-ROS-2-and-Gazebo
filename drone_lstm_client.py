# client.py
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from datetime import datetime

# === CONFIGURATION ===
SEQ_LENGTH = 20
BATCH_SIZE = 32
EPOCHS = 1

# === SET DRONE NAME AND DATA ===
DRONE_NAME = "drone1"  # Change this for other drones
DATA_FILE = f"/home/amine/Desktop/drones_data/{DRONE_NAME}/merged_labeled_file.csv"

# === LOAD DATA ===
df = pd.read_csv(DATA_FILE)
df = df.sort_values('Timestamp').reset_index(drop=True)

features = ['Distance (m)', 'RSSI (dBm)', 'SNIR (dB)',
            'total_received', 'drops_in_last_sec',
            'msgs_per_sec', 'loss_rate']
X_data = df[features].values
y_data = df['label'].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)

def create_sequences(data, labels, seq_length):
    sequences = []
    sequence_labels = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        label = labels[i + seq_length - 1]
        sequences.append(seq)
        sequence_labels.append(label)
    return np.array(sequences), np.array(sequence_labels)

X, y = create_sequences(X_scaled, y_data, SEQ_LENGTH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# === MODEL SETUP ===
model = Sequential([
    LSTM(64, input_shape=(SEQ_LENGTH, len(features)), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === FLWR CLIENT IMPLEMENTATION ===
class DroneClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        # Save locally after each round
        os.makedirs(f"saved_models/{DRONE_NAME}", exist_ok=True)
        round_number = config.get("server_round", 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"saved_models/{DRONE_NAME}/model_round{round_number}_{timestamp}.h5"
        model.save(save_path)
        print(f"✅ {DRONE_NAME} saved local model for round {round_number} at: {save_path}")

        return loss, len(X_test), {"accuracy": accuracy}

# === START CLIENT ===
fl.client.start_numpy_client(server_address="localhost:8080", client=DroneClient())

