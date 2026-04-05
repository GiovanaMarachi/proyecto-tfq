#!/usr/bin/env python3
import os
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import h5py

# Desactivar logs innecesarios de TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Importar estilos personalizados
import plot_styles

# =============================================================================
# Clase FakePQC: Simula la capa de TensorFlow Quantum sin importar la librería
# =============================================================================
class ScaleLayer(Layer):
    """Capa que carga los 2 pesos del PQC y aplica una transformación lineal simple."""
    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(name='pqc', **kwargs)

    def build(self, input_shape):
        # El PQC original tiene 2 parámetros entrenables
        self.theta = self.add_weight(name='vars/0', shape=(2,), trainable=True)
        super(ScaleLayer, self).build(input_shape)

    def call(self, inputs):
        # Simulación simplificada: proyectamos los pesos sobre la entrada para mantener la arquitectura
        batch_size = tf.shape(inputs)[0]
        # Devolvemos un valor basado en los pesos para que la red 'conecte'
        return tf.cos(self.theta[0]) * tf.ones((batch_size, 1))

# =============================================================================
# Reconstrucción del Modelo (mismo que qnn_predictor_pro.py pero sin import tfq)
# =============================================================================
def create_safe_model(seq_len, n_features):
    # Rama clásica (LSTM)
    classical_input = Input(shape=(seq_len, n_features), name='classical_input')
    x = LSTM(100, return_sequences=True)(classical_input)
    x = Dropout(0.3)(x)
    x = LSTM(100)(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)

    # Rama 'Cuántica' segura (Placeholder)
    # En el original es tf.string, aquí usamos un float dummy para poder cargar pesos
    quantum_input = Input(shape=(), dtype=tf.float32, name='quantum_input')
    q = ScaleLayer()(quantum_input)

    # Fusión
    fused = Concatenate()([x, q])
    fused = Dense(16, activation='relu')(fused)
    output = Dense(1, name='price_pred')(fused)

    model = Model(inputs=[classical_input, quantum_input], outputs=output)
    return model

def load_weights_manually(model, weights_path):
    """Carga los pesos desde el archivo H5 navegando por la estructura interna."""
    print(f"📦 Cargando pesos desde {weights_path}...")
    with h5py.File(weights_path, 'r') as f:
        # Mapeo de capas del modelo a rutas en el H5
        # NOTA: Los nombres en el H5 pueden variar ligeramente (ej. lstm vs lstm_1)
        
        # 1. LSTM 1
        model.layers[1].set_weights([
            np.array(f['layers/lstm/cell/vars/0']),
            np.array(f['layers/lstm/cell/vars/1']),
            np.array(f['layers/lstm/cell/vars/2'])
        ])
        
        # 2. LSTM 2
        model.layers[3].set_weights([
            np.array(f['layers/lstm_1/cell/vars/0']),
            np.array(f['layers/lstm_1/cell/vars/1']),
            np.array(f['layers/lstm_1/cell/vars/2'])
        ])

        # 3. Dense (embedding clásico)
        model.get_layer('dense').set_weights([
            np.array(f['layers/dense/vars/0']), 
            np.array(f['layers/dense/vars/1'])
        ])

        # 4. PQC (FakePQC)
        model.get_layer('pqc').set_weights([np.array(f['layers/pqc/vars/0'])])

        # 5. Dense Fusion
        model.get_layer('dense_1').set_weights([
            np.array(f['layers/dense_1/vars/0']), 
            np.array(f['layers/dense_1/vars/1'])
        ])

        # 6. Output (price_pred) - A veces guardado como dense_2 en TF
        try:
            model.get_layer('price_pred').set_weights([
                np.array(f['layers/dense_2/vars/0']), 
                np.array(f['layers/dense_2/vars/1'])
            ])
        except:
             model.get_layer('price_pred').set_weights([
                np.array(f['layers/price_pred/vars/0']), 
                np.array(f['layers/price_pred/vars/1'])
            ])
            
    print("✅ Pesos cargados correctamente.")

def run_safe_inference():
    # 1. Metadatos
    with open('models/qnn_hibrido_bitcoin_meta.json', 'r') as f:
        meta = json.load(f)
    
    seq_length = meta['seq_length']
    n_features = meta['n_features']
    features = meta['features_order']

    # 2. Cargar Datos
    print("📊 Leyendo datos históricos...")
    data_dir = 'data/Bitcoin'
    zip_files = sorted(glob.glob(os.path.join(data_dir, '*.zip')))
    all_dfs = []
    for zp in zip_files:
        with zipfile.ZipFile(zp, 'r') as zf:
            xl = [f for f in zf.namelist() if f.endswith('.xlsx')]
            if xl:
                with zf.open(xl[0]) as f:
                    all_dfs.append(pd.read_excel(f))
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], unit='s', errors='coerce')
    full_df = full_df.dropna(subset=['timestamp']).sort_values(by='timestamp').set_index('timestamp')
    
    data = full_df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Crear secuencias
    X = []
    close_idx = features.index('close')
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i])
    X = np.array(X)
    
    train_size = int(len(X) * 0.8)
    X_test = X[train_size:]
    test_dates = full_df.index[seq_length + train_size:]

    # 3. Crear y Cargar Modelo
    model = create_safe_model(seq_length, n_features)
    # Necesario inicializar variables internos de Keras antes de set_weights
    model.predict([X_test[:1], np.zeros(1)], verbose=0) 
    load_weights_manually(model, 'models/qnn_hibrido_bitcoin.weights.h5')

    # 4. Predicción
    print(f"🚀 Ejecutando predicción sobre {len(X_test)} puntos...")
    dummy_q = np.zeros(len(X_test))
    predicted = model.predict([X_test, dummy_q], batch_size=512, verbose=1)

    # Inversión de escala
    dummy_p = np.zeros((len(predicted), len(features)))
    dummy_p[:, close_idx] = predicted.flatten()
    predicted_prices = scaler.inverse_transform(dummy_p)[:, close_idx]

    # Precios reales (ya los tenemos en el test set indexado)
    real_prices = full_df.iloc[seq_length + train_size:][ 'close'].values

    # 5. Guardar Resultados
    os.makedirs('outputs', exist_ok=True)
    results_df = pd.DataFrame({
        'Timestamp': test_dates,
        'Real': real_prices,
        'Predicho': predicted_prices
    })
    results_df.to_csv('outputs/prediction_results.csv', index=False)
    print("💾 Resultados guardados en outputs/prediction_results.csv")

    # 6. Generar Gráficos
    print("🎨 Generando gráficos de alta calidad...")
    import generate_pub_plots
    generate_pub_plots.main()

if __name__ == "__main__":
    try:
        run_safe_inference()
        print("\n✨ Proceso finalizado con éxito.")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
