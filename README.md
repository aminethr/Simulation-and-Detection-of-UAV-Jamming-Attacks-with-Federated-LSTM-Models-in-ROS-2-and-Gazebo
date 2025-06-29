# 🛡️ Simulation and Detection of UAV Jamming Attacks with Federated LSTM Models in ROS 2 and Gazebo

This project simulates **normal** and **jammed** communication scenarios for UAVs using **ROS 2**, **Gazebo**, and **Scapy**, collects telemetry data (RSSI, SNIR, MAVLink packet stats), and applies **Federated Learning (FL)** with **LSTM models** to detect jamming attempts **without centralizing the raw data**.

---

## 📌 Project Overview

The simulation mimics a UAV navigating in a 3D Gazebo environment while its telemetry is monitored for signal degradation or interference.

The system:
- Simulates both normal and jammed conditions
- Monitors **RSSI** (Received Signal Strength Indicator) and **SNIR** (Signal to Noise plus Interference Ratio)
- Sniffs **MAVLink packets** to detect packet loss and drops
- Builds a **time-series dataset**
- Trains an **LSTM** model locally per drone
- Uses **Flower (FLWR)** to coordinate federated learning across multiple UAVs

---

## 🛰️ Simulation Modes

![Simulation Example](https://github.com/user-attachments/assets/4212d1c6-0d20-43cf-ade8-0c7531d8f9d9)

### 1. **Normal Mode**
In this mode, the UAV transmits telemetry over an undisturbed channel. The `rssi_simulator.py` calculates RSSI and SNIR using the logarithmic path loss model with:
- Moderate noise
- Low interference

### 2. **Jammed Mode**
Here, a **simulated jamming source** increases interference power. The `rssi_simulator.py` parameters are adjusted to simulate signal degradation:
- Strong interference (e.g., -45 dBm)
- RSSI drops
- SNIR becomes low (signal quality degrades)

---

## 📶 What are RSSI and SNIR?

- **RSSI (Received Signal Strength Indicator)**  
  A measure of signal power received by the antenna (in dBm).  
  Useful for estimating **distance** and **link quality**.

- **SNIR (Signal-to-Noise-and-Interference Ratio)**  
  Quantifies how strong the signal is compared to background noise and jamming interference.  
  Low SNIR typically means a jammed or degraded link.

---

## 📦 Packet Monitoring with Scapy

![Simulation2 Example](https://github.com/user-attachments/assets/4af0a18b-b8d2-4a84-a414-70312617b582)

We use a custom `packet_sniffer.py` script that:
- Sniffs incoming **MAVLink** packets on UDP port (e.g., 14550)
- Tracks:
  - Sequence numbers
  - Packet loss (drops)
  - Message rate (msgs/sec)
- Labels windows of time as **normal** or **jammed**
- Outputs features like:
  - `total_received`
  - `drops_in_last_sec`
  - `msgs_per_sec`
  - `loss_rate`
During jamming simulation, we flood the telemetry port (e.g., using ping -f or hping3) to induce packet loss and mimic real-world interference.
---

## 🧠 Dataset Generation

Each UAV collects and stores data to:
```text
~/Desktop/drones_data/droneX/{normal|jammed}/drone_rssi_log.csv

All signal + packet logs are merged and labeled using:

    label = 0 for normal intervals

    label = 1 for jammed intervals

The merged dataset is used to train a binary classifier (Normal vs Jammed).
```

## 🔁 Federated Learning Setup


We use Flower (flwr) to coordinate training without sharing raw data:

✅ Client (UAV)

Each UAV runs a client.py:

    Loads its own dataset (merged_labeled_file.csv)

    Preprocesses and standardizes input

    Trains a local LSTM model

    Sends model weights (not data) to server

    Saves its model after every round

fl.client.start_numpy_client(server_address="localhost:8080", client=DroneClient())

✅ Server

The central server runs server.py:

    Initializes global LSTM model

    Coordinates training with 2+ clients

    Aggregates model weights (FedAvg)

    Sends updated global model back after each round

📌 In this setup, no UAV sees another’s data, but all benefit from collaborative learning.

## 🧠 Model: LSTM for Time-Series

Each client trains an LSTM-based neural network on time windows (sequences of length 20):

model = Sequential([
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

Features:

    Distance (m)

    RSSI (dBm)

    SNIR (dB)

    total_received

    drops_in_last_sec

    msgs_per_sec

    loss_rate

## 📊 Results

![Results Example](https://github.com/user-attachments/assets/4c6c3103-8bff-4f97-bbc2-cc076bdd7e78)

Each round, clients log:

    Accuracy

    Loss

    Saved model path (per round)

The server prints aggregated accuracy.

## 🔧 How to Use

    Simulate normal and jammed data in Gazebo using ROS 2:

         Run rssi_simulator.py and packet_sniffer.py in both modes

    Label and merge the dataset per UAV using timestamp alignment between RSSI/SNIR logs and packet statistics.

    Start Flower server (server.py)

    Start multiple clients (client.py) with different DRONE_NAME and datasets
    
    After training, load the saved model (e.g., model_roundX_*.h5) in a ROS 2 Python node. Use it to classify live data collected during the mission:
    
    Continuously read telemetry (RSSI, SNIR, packet drops, etc.)
    Create sequences and pass them to the LSTM model
    If the predicted output exceeds a threshold (e.g., > 0.5), assume jamming is detected
    Trigger alerts or emergency drone actions (e.g., land or reroute)





