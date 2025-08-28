import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt

# Paso 1: Crea una capa cuántica parametrizada (PQC)
def create_qnn_layer():
    qubit = cirq.GridQubit(0, 0)
    theta = sympy.Symbol('theta')
    # Un circuito simple de rotación para la capa de entrenamiento
    pqc_circuit = cirq.Circuit(cirq.rz(theta)(qubit))
    return pqc_circuit, [qubit]

# Paso 2: Construye el modelo QNN híbrido con Keras
def create_model():
    # El modelo de Keras espera circuitos de codificación de datos
    data_input = tf.keras.Input(shape=(), dtype=tf.string)
    
    pqc_circuit, qubits = create_qnn_layer()
    
    # Crea una instancia de la capa AddCircuit con el PQC
    add_circuit_layer = tfq.layers.AddCircuit()
    
    # Llama a la capa con los datos de entrada y el circuito a añadir
    hybrid_circuit_output = add_circuit_layer(data_input, prepend=pqc_circuit)
    
    # La capa PQC ejecuta el circuito cuántico y mide el resultado
    qnn_output = tfq.layers.PQC(pqc_circuit, cirq.Z(qubits[0]))(hybrid_circuit_output)
    
    qnn_model = tf.keras.Model(inputs=data_input, outputs=qnn_output)
    return qnn_model

# Paso 3: Preprocesamiento de datos
# Crea datos de ejemplo para una función sinusoidal
X_train = np.arange(0, 10, 0.1)
y_train = np.sin(X_train)

# Convierte los valores de entrada clásicos en circuitos de Cirq
# para que el modelo pueda procesarlos.
def convert_to_circuits(values):
    qubit = cirq.GridQubit(0, 0)
    # Crea circuitos de codificación de datos para cada valor de entrada
    encoding_circuits = [cirq.Circuit(cirq.rx(value)(qubit)) for value in values]
    return tfq.convert_to_tensor(encoding_circuits)

# Convertir los datos de entrenamiento a circuitos cuánticos
X_train_circuits = convert_to_circuits(X_train)

# Paso 4: Preparar y entrenar el modelo
qnn_model = create_model()
qnn_model.compile(optimizer='adam', loss='mean_squared_error')

print("Comenzando el entrenamiento del modelo...")
history = qnn_model.fit(X_train_circuits, y_train, epochs=50, verbose=0)
print("Entrenamiento completado.")

# Paso 5: Visualizar la predicción
y_pred = qnn_model.predict(X_train_circuits)

plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, label='Valor Real (sin(x))')
plt.plot(X_train, y_pred, label='Predicción del Modelo')
plt.title('Predicción con una QNN')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('prediccion.png')
plt.close()