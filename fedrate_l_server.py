# server.py
import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import Dict, List, Tuple

# === CONFIGURATION ===
SEQ_LENGTH = 20
FEATURES_COUNT = 7

# === MODEL ARCHITECTURE ===
def create_model():
    model = Sequential([
        LSTM(64, input_shape=(SEQ_LENGTH, FEATURES_COUNT), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# === SERVER-SIDE DUMMY EVALUATION ===
def get_evaluate_fn(model):
    def evaluate(server_round: int, parameters: fl.common.NDArrays,
                 config: Dict[str, fl.common.Scalar]) -> Tuple[float, Dict[str, fl.common.Scalar]]:
        model.set_weights(parameters)
        return 0.0, {"accuracy": 0.0}
    return evaluate

# === AGGREGATED CLIENT METRICS ===
def aggregate_evaluate(metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
    print(f"🔍 Aggregated accuracy: {weighted_accuracy:.4f}")
    return {"accuracy": weighted_accuracy}

# === STRATEGY WITHOUT SAVING ===
class NoSaveFedAvgStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        result = super().aggregate_fit(server_round, results, failures)
        return result

# === MAIN SERVER LOGIC ===
if __name__ == "__main__":
    model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(model.get_weights())

    strategy = NoSaveFedAvgStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(model),
        evaluate_metrics_aggregation_fn=aggregate_evaluate,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

