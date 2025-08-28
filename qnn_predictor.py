# qnn_hibrido_bitcoin.py
# Modelo híbrido LSTM + QNN con TFQ siguiendo la estructura de tu ANN original,
# con guardado robusto (pesos .h5 + checkpoint + metadatos) para evitar problemas de serialización con TFQ.

import os
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# --- Cuántico (TFQ) ---
import tensorflow_quantum as tfq
import cirq
import sympy

# =========================
# Paso 1: Cargar datos reales de Bitcoin desde el ZIP (XLSX)
# =========================
zip_file_path = 'data/20241201.zip'
excel_file_in_zip = '20241201.xlsx'

try:
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        with zf.open(excel_file_in_zip) as f:
            df = pd.read_excel(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo ZIP en la ruta: {zip_file_path}")
    print("Asegúrate de que '20241201.zip' esté en la carpeta 'data/'.")
    raise SystemExit(1)
except KeyError:
    print(f"Error: No se encontró el archivo '{excel_file_in_zip}' dentro del ZIP.")
    print(f"Asegúrate de que el nombre del archivo dentro del ZIP sea exactamente '{excel_file_in_zip}'.")
    raise SystemExit(1)
except Exception as e:
    print(f"Ocurrió un error al leer el archivo Excel: {e}")
    print("Asegúrate de tener 'openpyxl' instalado: pip install openpyxl")
    raise SystemExit(1)

# Convertir timestamp y setear índice (opcional)
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.set_index('timestamp')

# Selección de features
features = ['open', 'high', 'low', 'close', 'basevolume', 'usdtvolume']
for col in features:
    if col not in df.columns:
        raise ValueError(f"Falta la columna '{col}' en el Excel leído.")

data = df[features].values

# =========================
# Paso 2: Normalización
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# =========================
# Paso 3: Crear secuencias (igual que tu ANN)
# =========================
def create_sequences(data, seq_len):
    X, y = [], []
    close_idx = features.index('close')
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])              # (seq_len, n_features)
        y.append(data[i, close_idx])               # valor normalizado de 'close' actual
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(scaled_data, seq_length)
print("Forma de X (secuencias, timesteps, features):", X.shape)
print("Forma de y (salida):", y.shape)

# =========================
# Paso 4: Train / Test split
# =========================
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# =========================
# Utilidades TFQ: circuitos por muestra
# =========================
# Usaremos 2 qubits y codificaremos:
# - Qubit 0: 'close' del último timestep de la secuencia.
# - Qubit 1: 'usdtvolume' del último timestep.
# Rotaciones Rx(angle) con angle = valor_normalizado * pi, más un CZ para entrelazar.

def build_sample_circuit(last_close_norm, last_usdtvol_norm):
    q0, q1 = cirq.GridQubit.rect(1, 2)
    circuit = cirq.Circuit()
    theta0 = float(last_close_norm) * np.pi
    theta1 = float(last_usdtvol_norm) * np.pi
    circuit.append(cirq.rx(theta0)(q0))
    circuit.append(cirq.rx(theta1)(q1))
    circuit.append(cirq.CZ(q0, q1))
    return circuit

def sequences_to_circuits(X_seq):
    close_idx = features.index('close')
    usdt_idx = features.index('usdtvolume')
    circuits = []
    for sample in X_seq:
        last_close = sample[-1, close_idx]
        last_usdtvol = sample[-1, usdt_idx]
        circuits.append(build_sample_circuit(last_close, last_usdtvol))
    return tfq.convert_to_tensor(circuits)

X_train_circuits = sequences_to_circuits(X_train)
X_test_circuits  = sequences_to_circuits(X_test)

# =========================
# Paso 5: Crear modelo híbrido LSTM + QNN
# =========================
def create_hybrid_lstm_qnn_model(seq_len, n_features):
    # Rama clásica (LSTM)
    classical_input = Input(shape=(seq_len, n_features), name='classical_input')
    x = LSTM(100, return_sequences=True)(classical_input)
    x = Dropout(0.2)(x)
    x = LSTM(100)(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)  # embedding clásico

    # Rama cuántica (TFQ): input de circuitos (tf.string)
    quantum_input = Input(shape=(), dtype=tf.string, name='quantum_input')

    # Circuito parametrizado de 2 qubits para PQC
    q0, q1 = cirq.GridQubit.rect(1, 2)
    theta0 = sympy.Symbol('theta0')
    theta1 = sympy.Symbol('theta1')
    pqc_circuit = cirq.Circuit(
        cirq.ry(theta0)(q0),
        cirq.ry(theta1)(q1),
        cirq.CZ(q0, q1)
    )
    readout = cirq.Z(q0)

    pqc_layer = tfq.layers.PQC(pqc_circuit, readout, name='pqc')
    q = pqc_layer(quantum_input)  # (batch, 1)

    # Fusión de ramas
    fused = Concatenate()([x, q])
    fused = Dense(16, activation='relu')(fused)
    output = Dense(1, name='price_pred')(fused)

    model = Model(inputs=[classical_input, quantum_input], outputs=output)
    return model

model = create_hybrid_lstm_qnn_model(seq_length, X.shape[2])
model.compile(optimizer='adam', loss='mean_squared_error')

print(model.summary())

# =========================
# Paso 6: Entrenar
# =========================
history = model.fit(
    [X_train, X_train_circuits],
    y_train,
    epochs=50,
    batch_size=64,
    validation_data=([X_test, X_test_circuits], y_test),
    verbose=1
)

# =========================
# Paso 7: Predicción (antes de guardar)
# =========================
predicted = model.predict([X_test, X_test_circuits])

# Inversión de escala solo para 'close'
close_idx = features.index('close')
dummy_pred = np.zeros((len(predicted), len(features)))
dummy_pred[:, close_idx] = predicted.flatten()
predicted_prices = scaler.inverse_transform(dummy_pred)[:, close_idx]

dummy_real = np.zeros((len(y_test), len(features)))
dummy_real[:, close_idx] = y_test.flatten()
real_prices = scaler.inverse_transform(dummy_real)[:, close_idx]

# =========================
# Paso 8: Visualizaciones y guardado en outputs/
# =========================
os.makedirs('outputs', exist_ok=True)

plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Precio Real')
plt.plot(predicted_prices, label='Precio Predicho')
plt.title('Predicción de Precios de Bitcoin (Híbrido LSTM+QNN)')
plt.xlabel('Puntos de Datos')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/bitcoin_qnn_prediction_real_data_plot.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo durante el Entrenamiento (LSTM+QNN)')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/bitcoin_qnn_training_loss_plot.png')
plt.close()

# =========================
# Paso 9: Guardado robusto (TFQ)
# =========================
os.makedirs('models', exist_ok=True)

# (A) Guardar SOLO pesos Keras (recomendado)
model.save_weights('models/qnn_hibrido_bitcoin.weights.h5')

# (B) Guardar checkpoint TensorFlow (reconstruible)
ckpt = tf.train.Checkpoint(model=model)
ckpt_path = ckpt.save('models/qnn_hibrido_bitcoin_ckpt')
# Nota: se guardará con sufijo incremental, p. ej., ...-1, ...-2, etc.

# (C) Guardar metadatos mínimos para reconstruir
meta = {
    "seq_length": int(X.shape[1]),
    "n_features": int(X.shape[2]),
    "features_order": features
}
with open('models/qnn_hibrido_bitcoin_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("✅ Listo.")
print("   - Pesos .h5: models/qnn_hibrido_bitcoin.weights.h5")
print("   - Checkpoint (prefijo): models/qnn_hibrido_bitcoin_ckpt-*")
print("   - Metadatos: models/qnn_hibrido_bitcoin_meta.json")
print("   - Gráficos: outputs/bitcoin_qnn_prediction_real_data_plot.png y outputs/bitcoin_qnn_training_loss_plot.png")
