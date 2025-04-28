import numpy as np
import matplotlib.pyplot as plt

# Datos de entrada
X = np.array([1, 2, 3, 4, 5])  # Horas de estudio
Y = np.array([1, 2, 2.8, 4.1, 5.1])  # Puntaje en el examen

# Número de datos
n = len(X)

# Calcular las sumas necesarias para las fórmulas
sum_X = np.sum(X)
sum_Y = np.sum(Y)
sum_XY = np.sum(X * Y)
sum_X2 = np.sum(X ** 2)

# Calcular los coeficientes de la recta
beta_1 = (n * sum_XY - sum_X * sum_Y) / (n * sum_X2 - sum_X ** 2)
beta_0 = (sum_Y - beta_1 * sum_X) / n

# Mostrar los coeficientes
print(f'Coeficiente beta_1 (pendiente): {beta_1}')
print(f'Coeficiente beta_0 (intercepto): {beta_0}')

# Predicciones utilizando el modelo
Y_pred = beta_0 + beta_1 * X

# Graficar los datos y la línea de regresión
plt.scatter(X, Y, color='blue', label='Datos reales')
plt.plot(X, Y_pred, color='red', label='Línea de regresión')
plt.xlabel('Horas de Estudio')
plt.ylabel('Puntaje')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.show()

# Mostrar las predicciones
print(f'Predicciones: {Y_pred}')