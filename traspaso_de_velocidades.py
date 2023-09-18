import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

kilometros_hora = np.array([1, 10, 12, 18, 22, 25, 30], dtype=float)
nudos = np.array([0.53, 5.39, 6.47, 9.71, 11.87, 13.49, 16.19], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])

modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(kilometros_hora, nudos, epochs=2000, verbose=False)
print("Modelo entrenado..!!")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])


print("Hagamos una prediccion.!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " nudos..!")

print("variables internas del modelo")
print(capa.get_weights())
