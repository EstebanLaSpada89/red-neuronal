import tensorflow as tf
import numpy as np

# colocamos un arreglo de numeros con nuestras entradas en grados celsius

celsius = np.array ([-40, -10, 0, 8, 15, 22, 38], dtype=float)

# colocamos los resultados de nuestras entradas en grados fahrenheit

fahrenheit = np.array ([-40, 14, 32, 46, 59, 72, 100], dtype=float) #estos son los ejemplos que la red neuronal usara para aprender

# armamos las capas de la red neuronal con "keras"

capa = tf.keras.layers.Dense (units=1, input_shape=[1]) # tf.keras.layers.Dense , se crea una capa densa, las capas densas tienen conecciones desde cada neurona hacia todas las neuronas de las siguiente capa
# units=1 , se le indica las unidades o neuronas de la capa
# input_shape=[1] aca le decimos que tenemos una entrada con una neurona

modelo = tf.keras.Sequential([capa]) # se le indica al modelo la capa creada

modelo.compile (
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# optimizer=tf.keras.optimizers.Adam() le permite saber como ajustar los pesos y sesgos de manera eficiente para que aprenda y no para que desaprenda, osea para que poco a poco vaya mejorando en el aprendizaje, el valor (0.1) es el valor de la tasa de aprendizaje
# loss='mean_squared_error'  es la funcion de perdida


# Comenzamos el entrenamiento de la red neuronal

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=2000, verbose=False)# le damos el dato de entrada celsius y el resultado esperado osea fahrenheit, ademas le decimos cuantas vueltas queremos que lo intente
print("Modelo entrenado..!!")


# Creamos la funcion de perdida, esto nos indica que tan mal esta el resultado de la red en cada vuelta que de

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel ("Magnitud de perdida")
plt.plot(historial.history["loss"])


print("Hagamos una prediccion.!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " Fahrenheit..!")

# estructura interna de la red, datos asignados
print("variables internas del modelo")
print(capa.get_weights())
