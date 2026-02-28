#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import plot_styles
from datetime import datetime

# Desactivar logs innecesarios
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =========================
# Capa Cuántica Simulada (Segura)
# =========================
class FakePQC(Layer):
    def __init__(self, **kwargs):
        super(FakePQC, self).__init__(name='pqc', **kwargs)

    def build(self, input_shape):
        self.theta = self.add_weight(name='vars/0', shape=(2,), trainable=True,
                                     initializer='random_normal')
        super(FakePQC, self).build(input_shape)

    def call(self, inputs):
        # Ahora el aporte cuántico DEPENDE de la entrada real (inputs)
        # Multiplicamos la entrada por theta para simular un procesamiento cuántico
        batch_size = tf.shape(inputs)[0]
        # Aseguramos que inputs tenga forma (batch, 1) y operamos
        inputs_reshaped = tf.reshape(inputs, [batch_size, 1])
        return tf.cos(inputs_reshaped * self.theta[0] + self.theta[1])

# =========================
# Cálculo de RSI (Indicador Técnico)
# =========================
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================
# Carga y Preprocesamiento de Datos
# =========================
def load_optimized_data(csv_path):
    print(f"📊 Cargando y optimizando datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    
    # Agregamos RSI para ayudar al modelo con el momentum
    df['btc_rsi'] = calculate_rsi(df['btc_close'])
    df['eth_rsi'] = calculate_rsi(df['eth_close'])
    
    # Rellenar NaNs del RSI inicial
    df = df.fillna(method='bfill')
    
    features = [
        'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_basevolume', 'btc_usdtvolume',
        'eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_basevolume', 'eth_usdtvolume',
        'btc_rsi', 'eth_rsi'
    ]
    
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler, features, df.index

def create_sequences(data, seq_len, target_idx):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)

# =========================
# Arquitectura del Modelo (Optimizada)
# =========================
def create_model(seq_len, n_features):
    classical_input = Input(shape=(seq_len, n_features), name='classical_input')
    
    # Reducción de capacidad y L2 para evitar lag y memorización
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(classical_input)
    x = Dropout(0.3)(x)
    x = LSTM(32, kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)

    quantum_input = Input(shape=(1,), name='quantum_input') # Ahora recibe 1 dato
    q = FakePQC()(quantum_input)

    # Fusión donde el QNN aporta una corrección no lineal basada en el precio reciente
    fused = Concatenate()([x, q])
    fused = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(fused)
    output = Dense(1, name='price_pred')(fused)

    model = Model(inputs=[classical_input, quantum_input], outputs=output)
    return model

# =========================
# Script Principal
# =========================
def main():
    csv_path = 'data/datos_btc_eth_unificados.csv'
    seq_length = 40 # Aumentamos a 40 para más contexto y reducir lag
    epochs = 20
    batch_size = 64

    # 1. Datos
    scaled_data, scaler, features, timestamps = load_optimized_data(csv_path)
    target_idx = features.index('btc_close')
    X, y = create_sequences(scaled_data, seq_length, target_idx)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = timestamps[seq_length + train_size:]

    # 2. Modelo
    model = create_model(seq_length, len(features))
    # Learning rate más bajo (0.0001) para mayor estabilidad, como el pro
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    
    # Callbacks: Early Stopping + ReduceLROnPlateau para convergencia fina
    print(f"🚀 Iniciando Prueba Fase 2 Estricta ({epochs} épocas)...")
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )
    
    # Aporte QNN: Pasamos el último precio de cierre 'btc_close' de la secuencia
    close_idx_in_features = features.index('btc_close')
    last_prices_train = X_train[:, -1, close_idx_in_features]
    last_prices_test = X_test[:, -1, close_idx_in_features]
    
    history = model.fit(
        [X_train, last_prices_train], y_train,
        validation_data=([X_test, last_prices_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # 4. Guardado y Predicción
    os.makedirs('models', exist_ok=True)
    model.save_weights('models/qnn_hibrido_fase2.weights.h5')
    
    predicted = model.predict([X_test, last_prices_test], batch_size=512)
    
    dummy_p = np.zeros((len(predicted), len(features)))
    dummy_p[:, target_idx] = predicted.flatten()
    predicted_prices = scaler.inverse_transform(dummy_p)[:, target_idx]

    dummy_r = np.zeros((len(y_test), len(features)))
    dummy_r[:, target_idx] = y_test.flatten()
    real_prices = scaler.inverse_transform(dummy_r)[:, target_idx]

    os.makedirs('outputs', exist_ok=True)
    csv_out = 'outputs/prediction_results_final.csv'
    pd.DataFrame({
        'Timestamp': test_dates,
        'Real': real_prices,
        'Predicho': predicted_prices
    }).to_csv(csv_out, index=False)

    # 5. Generar Reporte Final
    print("🎨 Generando Gráficas de Máxima Calidad...")
    import generate_pub_plots
    hist_df = pd.DataFrame(history.history)
    report_path = generate_pub_plots.generate_all_plots(csv_out, history_df=hist_df)
    
    print(f"\n✨ PRUEBA FINAL COMPLETADA. Reporte en: {report_path}")

if __name__ == "__main__":
    main()
