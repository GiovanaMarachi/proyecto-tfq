# qnn_predictor_pro.py
# Modelo híbrido LSTM + QNN que predice el precio de BTC usando datos históricos de BTC y ETH.

import os
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# --- Cuántico (TFQ) ---
import tensorflow_quantum as tfq
import cirq
import sympy

# =========================
# Paso 1: Carga y Fusión de Datos (BTC y ETH)
# =========================
def load_crypto_data(data_dir, prefix):
    """Carga todos los archivos de datos de una cripto desde un directorio y les añade un prefijo."""
    zip_files = sorted(glob.glob(os.path.join(data_dir, '*.zip')))
    if not zip_files:
        print(f"Advertencia: No se encontraron archivos .zip en la ruta: {data_dir}")
        return pd.DataFrame()

    all_dfs = []
    for zip_file_path in zip_files:
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zf:
                excel_files = [f for f in zf.namelist() if f.endswith('.xlsx')]
                if not excel_files:
                    continue
                with zf.open(excel_files[0]) as f:
                    df = pd.read_excel(f)
                    # Renombrar columnas con prefijo
                    df.columns = [f"{prefix}_{col.lower()}" for col in df.columns]
                    all_dfs.append(df)
        except Exception as e:
            print(f"Error procesando {zip_file_path}: {e}")
            raise SystemExit(1)

    if not all_dfs:
        return pd.DataFrame()

    full_df = pd.concat(all_dfs, ignore_index=True)
    timestamp_col = f'{prefix}_timestamp'
    if timestamp_col in full_df.columns:
        full_df[timestamp_col] = pd.to_datetime(full_df[timestamp_col], unit='s', errors='coerce')
        full_df = full_df.dropna(subset=[timestamp_col])
        full_df = full_df.sort_values(by=timestamp_col).set_index(timestamp_col)
    return full_df

# Cargar datos de BTC y ETH
btc_df = load_crypto_data('data/Bitcoin', 'btc')
eth_df = load_crypto_data('data/Ethereum', 'eth')

if btc_df.empty or eth_df.empty:
    raise ValueError("No se pudieron cargar los datos de una o ambas criptomonedas.")

# Fusionar DataFrames en base al timestamp
full_df = pd.merge(btc_df, eth_df, left_index=True, right_index=True, how='outer')
# Rellenar valores faltantes (si los hay) y luego eliminar filas con NaN restantes
full_df = full_df.ffill().dropna()

# Selección de features (12 en total)
features = [
    'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_basevolume', 'btc_usdtvolume',
    'eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_basevolume', 'eth_usdtvolume'
]
for col in features:
    if col not in full_df.columns:
        raise ValueError(f"Falta la columna '{col}' en los datos fusionados.")

data = full_df[features].values

# =========================
# Paso 2: Normalización
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# =========================
# Paso 3: Crear secuencias
# =========================
def create_sequences(data, seq_len):
    X, y = [], []
    # El objetivo sigue siendo predecir el precio de cierre de BTC
    btc_close_idx = features.index('btc_close')
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, btc_close_idx])
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
# Utilidades TFQ: circuitos multivariable
# =========================
def build_sample_circuit(btc_close, btc_vol, eth_close, eth_vol):
    qubits = cirq.GridQubit.rect(1, 4)
    circuit = cirq.Circuit()
    # Codificar los 4 valores en rotaciones de los 4 qubits
    circuit.append(cirq.rx(float(btc_close) * np.pi)(qubits[0]))
    circuit.append(cirq.rx(float(btc_vol) * np.pi)(qubits[1]))
    circuit.append(cirq.rx(float(eth_close) * np.pi)(qubits[2]))
    circuit.append(cirq.rx(float(eth_vol) * np.pi)(qubits[3]))
    # Entrelazar los qubits
    circuit.append(cirq.CZ(qubits[0], qubits[1]))
    circuit.append(cirq.CZ(qubits[1], qubits[2]))
    circuit.append(cirq.CZ(qubits[2], qubits[3]))
    return circuit

def sequences_to_circuits(X_seq):
    btc_close_idx = features.index('btc_close')
    btc_vol_idx = features.index('btc_usdtvolume')
    eth_close_idx = features.index('eth_close')
    eth_vol_idx = features.index('eth_usdtvolume')
    circuits = []
    for sample in X_seq:
        # Extraer el último valor de cada feature relevante
        last_btc_close = sample[-1, btc_close_idx]
        last_btc_vol = sample[-1, btc_vol_idx]
        last_eth_close = sample[-1, eth_close_idx]
        last_eth_vol = sample[-1, eth_vol_idx]
        circuits.append(build_sample_circuit(last_btc_close, last_btc_vol, last_eth_close, last_eth_vol))
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
    x = Dropout(0.3)(x)
    x = LSTM(100)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)

    # Rama cuántica (TFQ)
    quantum_input = Input(shape=(), dtype=tf.string, name='quantum_input')
    # PQC con 4 qubits y 4 parámetros entrenables
    qubits = cirq.GridQubit.rect(1, 4)
    params = sympy.symbols('theta0:4')
    pqc_circuit = cirq.Circuit(
        cirq.ry(params[0])(qubits[0]),
        cirq.ry(params[1])(qubits[1]),
        cirq.ry(params[2])(qubits[2]),
        cirq.ry(params[3])(qubits[3]),
        cirq.CZ(qubits[0], qubits[1]),
        cirq.CZ(qubits[1], qubits[2]),
        cirq.CZ(qubits[2], qubits[3])
    )
    readout = cirq.Z(qubits[0])

    pqc_layer = tfq.layers.PQC(pqc_circuit, readout, name='pqc')
    q = pqc_layer(quantum_input)

    # Fusión de ramas
    fused = Concatenate()([x, q])
    fused = Dense(16, activation='relu')(fused)
    output = Dense(1, name='price_pred')(fused)

    model = Model(inputs=[classical_input, quantum_input], outputs=output)
    return model

model = create_hybrid_lstm_qnn_model(seq_length, X.shape[2])
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
print(model.summary())

# =========================
# Paso 6: Entrenar
# =========================
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    [X_train, X_train_circuits],
    y_train,
    epochs=200,
    batch_size=64,
    validation_data=([X_test, X_test_circuits], y_test),
    callbacks=[early_stopping],
    verbose=1
)

# =========================
# Paso 7: Predicción
# =========================
predicted_scaled = model.predict([X_test, X_test_circuits])

# Inversión de escala para el precio de BTC
btc_close_idx = features.index('btc_close')
dummy_pred = np.zeros((len(predicted_scaled), len(features)))
dummy_pred[:, btc_close_idx] = predicted_scaled.flatten()
predicted_prices = scaler.inverse_transform(dummy_pred)[:, btc_close_idx]

dummy_real = np.zeros((len(y_test), len(features)))
dummy_real[:, btc_close_idx] = y_test.flatten()
real_prices = scaler.inverse_transform(dummy_real)[:, btc_close_idx]

# =========================
# Paso 7.1: Análisis de Métricas y Rango de Error
# Este bloque calcula métricas de error comunes (RMSE, MAE, MAPE) y también
# identifica los puntos de datos con el mayor y menor error.
# =========================
# Calcular y mostrar métricas de error
mae = mean_absolute_error(real_prices, predicted_prices)
mape = mean_absolute_percentage_error(real_prices, predicted_prices)
mse = mean_squared_error(real_prices, predicted_prices)
rmse = np.sqrt(mse)

print("\n" + "="*25)
print("Métricas de Error y Precisión del Modelo")
print("="*25)
print(f"   - MAE (Error Absoluto Medio):      ${mae:.2f}")
print(f"   - MAPE (Error Porcentual Absoluto Medio): {mape:.2%}")
print(f"   - RMSE (Raíz del Error Cuadrático Medio): ${rmse:.2f}")
print(f"   - Precisión del Modelo (1 - MAPE):   {1 - mape:.2%}")
print("="*25)

# Calcular el error absoluto entre el precio real y el predicho
error_absoluto = np.abs(real_prices - predicted_prices)

# Encontrar el índice del error mínimo y máximo
min_error_idx = np.argmin(error_absoluto)
max_error_idx = np.argmax(error_absoluto)

# Obtener los valores para la mejor y peor predicción
mejor_prediccion = {
    "real": real_prices[min_error_idx],
    "predicho": predicted_prices[min_error_idx],
    "error": error_absoluto[min_error_idx]
}
peor_prediccion = {
    "real": real_prices[max_error_idx],
    "predicho": predicted_prices[max_error_idx],
    "error": error_absoluto[max_error_idx]
}

print("\n" + "="*25)
print("Análisis de Rango de Error Individual")
print("="*25)
print(f"Mejor Predicción (Error Mínimo):")
print(f"   - Precio Real:      ${mejor_prediccion['real']:.2f}")
print(f"   - Precio Predicho:    ${mejor_prediccion['predicho']:.2f}")
print(f"   - Error Absoluto:     ${mejor_prediccion['error']:.2f}")
print("-" * 25)
print(f"Peor Predicción (Error Máximo):")
print(f"   - Precio Real:      ${peor_prediccion['real']:.2f}")
print(f"   - Precio Predicho:    ${peor_prediccion['predicho']:.2f}")
print(f"   - Error Absoluto:     ${peor_prediccion['error']:.2f}")
print("="*25 + "\n")


# =========================
# Paso 8: Visualizaciones y guardado
# =========================
os.makedirs('outputs', exist_ok=True)

# --- Gráfico 1: Vista General ---
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Precio Real de BTC')
plt.plot(predicted_prices, label='Precio Predicho de BTC')
plt.title('Predicción de Precios de Bitcoin con datos de ETH (Híbrido LSTM+QNN)')
plt.xlabel('Puntos de Datos')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/bitcoin_multivariable_qnn_prediction_plot.png')
plt.close()

# --- Gráfico 2: Zoom en la Mejor Predicción ---
context_window = 20
start_idx = max(0, min_error_idx - context_window)
end_idx = min(len(real_prices), min_error_idx + context_window)

plt.figure(figsize=(12, 6))
plt.plot(range(start_idx, end_idx), real_prices[start_idx:end_idx], label='Precio Real de BTC', color='blue', marker='o', linestyle='-')
plt.plot(range(start_idx, end_idx), predicted_prices[start_idx:end_idx], label='Precio Predicho de BTC', color='red', marker='x', linestyle='--')
plt.axvline(x=min_error_idx, color='green', linestyle='--', label=f'Mejor Predicción (Error: ${mejor_prediccion["error"]:.2f})')
plt.title('Vista Ampliada: Mejor Predicción (Error Mínimo)')
plt.xlabel('Puntos de Datos (Índice)')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/bitcoin_qnn_mejor_prediccion_zoom.png')
plt.close()

# --- Gráfico 3: Zoom en la Peor Predicción ---
start_idx = max(0, max_error_idx - context_window)
end_idx = min(len(real_prices), max_error_idx + context_window)

plt.figure(figsize=(12, 6))
plt.plot(range(start_idx, end_idx), real_prices[start_idx:end_idx], label='Precio Real de BTC', color='blue', marker='o', linestyle='-')
plt.plot(range(start_idx, end_idx), predicted_prices[start_idx:end_idx], label='Precio Predicho de BTC', color='red', marker='x', linestyle='--')
plt.axvline(x=max_error_idx, color='purple', linestyle='--', label=f'Peor Predicción (Error: ${peor_prediccion["error"]:.2f})')
plt.title('Vista Ampliada: Peor Predicción (Error Máximo)')
plt.xlabel('Puntos de Datos (Índice)')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/bitcoin_qnn_peor_prediccion_zoom.png')
plt.close()


plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo durante el Entrenamiento (LSTM+QNN Optimizado)')
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
print("   - Gráficos: outputs/bitcoin_multivariable_qnn_prediction_plot.png, outputs/bitcoin_qnn_mejor_prediccion_zoom.png, outputs/bitcoin_qnn_peor_prediccion_zoom.png, outputs/bitcoin_qnn_training_loss_plot.png")

print("\nFin del script.")