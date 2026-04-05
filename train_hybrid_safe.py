#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import plot_styles

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
        batch_size = tf.shape(inputs)[0]
        # Simulación: El peso influye en una salida constante por batch
        return tf.cos(self.theta[0]) * tf.ones((batch_size, 1))

# =========================
# Carga y Preprocesamiento de Datos
# =========================
def load_unified_data(csv_path):
    print(f"📊 Cargando datos unificados desde {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').set_index('timestamp')
    
    # Todas las columnas excepto timestamp son features (12 en total)
    features = [
        'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_basevolume', 'btc_usdtvolume',
        'eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_basevolume', 'eth_usdtvolume'
    ]
    
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler, features, df.index

def create_sequences(data, seq_len, target_idx):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, target_idx]) # Predecimos btc_close
    return np.array(X), np.array(y)

# =========================
# Arquitectura del Modelo
# =========================
def create_model(seq_len, n_features):
    # Rama Clásica
    classical_input = Input(shape=(seq_len, n_features), name='classical_input')
    x = LSTM(100, return_sequences=True)(classical_input)
    x = Dropout(0.3)(x)
    x = LSTM(100)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)

    # Rama Cuántica Simulada
    quantum_input = Input(shape=(), dtype=tf.float32, name='quantum_input')
    q = FakePQC()(quantum_input)

    # Fusión
    fused = Concatenate()([x, q])
    fused = Dense(16, activation='relu')(fused)
    output = Dense(1, name='price_pred')(fused)

    model = Model(inputs=[classical_input, quantum_input], outputs=output)
    return model

# =========================
# Script Principal de Entrenamiento
# =========================
def main():
    csv_path = 'data/datos_btc_eth_unificados.csv'
    seq_length = 20
    epochs = 30 # Reducido para rapidez en esta prueba, puedes aumentarlo
    batch_size = 64

    # 1. Preparar datos
    scaled_data, scaler, features, timestamps = load_unified_data(csv_path)
    target_idx = features.index('btc_close')
    
    X, y = create_sequences(scaled_data, seq_length, target_idx)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = timestamps[seq_length + train_size:]

    # 2. Construir modelo
    model = create_model(seq_length, len(features))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # 3. Entrenar con Early Stopping
    print(f"🚀 Iniciando entrenamiento ({epochs} épocas con Early Stopping)...")
    dummy_input_train = np.zeros(len(X_train))
    dummy_input_test = np.zeros(len(X_test))
    
    # Callback para detener el entrenamiento si la val_loss deja de mejorar
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        [X_train, dummy_input_train], y_train,
        validation_data=([X_test, dummy_input_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # 4. Guardar resultados
    print("💾 Guardando modelo y metadatos...")
    os.makedirs('models', exist_ok=True)
    model.save_weights('models/qnn_hibrido_bitcoin.weights.h5')
    
    meta = {
        "seq_length": seq_length,
        "n_features": len(features),
        "features_order": features
    }
    with open('models/qnn_hibrido_bitcoin_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # 5. Predicción y CSV
    print("📈 Generando predicciones finales...")
    predicted = model.predict([X_test, dummy_input_test], batch_size=512)
    
    # Invertir escala para el reporte
    dummy_p = np.zeros((len(predicted), len(features)))
    dummy_p[:, target_idx] = predicted.flatten()
    predicted_prices = scaler.inverse_transform(dummy_p)[:, target_idx]

    dummy_r = np.zeros((len(y_test), len(features)))
    dummy_r[:, target_idx] = y_test.flatten()
    real_prices = scaler.inverse_transform(dummy_r)[:, target_idx]

    os.makedirs('outputs', exist_ok=True)
    pd.DataFrame({
        'Timestamp': test_dates,
        'Real': real_prices,
        'Predicho': predicted_prices
    }).to_csv('outputs/prediction_results.csv', index=False)

    # 6. Gráficos detallados con historial
    print("🎨 Generando gráficas avanzadas...")
    import generate_pub_plots
    hist_df = pd.DataFrame(history.history)
    report_path = generate_pub_plots.generate_all_plots('outputs/prediction_results.csv', history_df=hist_df)
    
    print(f"\n✅ Proceso completado. Gráficas guardadas en: {report_path}")

if __name__ == "__main__":
    main()
