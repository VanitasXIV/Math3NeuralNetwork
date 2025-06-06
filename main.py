# ================================================
# PARTE 1 - Análisis Exploratorio y Transformaciones
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# ------------------------------------------------
# Paso 0: Cargar el dataset
# ------------------------------------------------
df = pd.read_csv("video games sales.csv")

# ------------------------------------------------
# Paso 1: Crear columna objetivo
# ------------------------------------------------
df['Exito_Ventas'] = (df['Global_Sales'] >= 1.0).astype(int)

# ------------------------------------------------
# Paso 2: Detección de Outliers (Método IQR)
# ------------------------------------------------
sales_columns = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
outliers_report = {}

for col in sales_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_report[col] = len(outliers)

print("Cantidad de outliers detectados por el método IQR:")
for col, count in outliers_report.items():
    print(f"{col}: {count} casos")

# Boxplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
axes = axes.flatten()
for i, col in enumerate(sales_columns):
    sns.boxplot(data=df, x=col, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Boxplot - {col}')
if len(sales_columns) < len(axes):
    axes[-1].axis('off')
plt.tight_layout()
plt.show()

# ------------------------------------------------
# Paso 3: Matriz de Correlación
# ------------------------------------------------
numeric_cols = ['Rank', 'Year'] + sales_columns + ['Exito_Ventas']
df_numeric = df[numeric_cols].dropna()

correlation_matrix = df_numeric.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlación entre Variables Numéricas")
plt.tight_layout()
plt.show()

print("Correlación con 'Exito_Ventas':")
correlation_with_target = correlation_matrix['Exito_Ventas'].sort_values(ascending=False)
print(correlation_with_target)

# ------------------------------------------------
# Paso 4: Transformaciones Preliminares
# ------------------------------------------------

# Normalización usando la desviación estándar (Z-score)
df[sales_columns] = (df[sales_columns] - df[sales_columns].mean()) / df[sales_columns].std()
df = pd.get_dummies(df, columns=['Platform', 'Genre', 'Publisher'], drop_first=True)

# ================================================
# PARTE 2 - Red Neuronal en NumPy
# ================================================

X = df.drop(columns=['Name', 'Rank', 'Exito_Ventas', 'Year'])
y = df['Exito_Ventas'].values.reshape(-1, 1)
#Función de scikit-learn para dividir el dataset en entrenamiento y validación
#Convierte el DataFrame a un array de NumPy
#El 20% de los datos se utiliza para validación
#El 80% de los datos se utiliza para entrenamiento
#El parámetro random_state asegura que la división sea reproducible
#El parámetro test_size indica el tamaño del conjunto de validación
X_train, X_val, y_train, y_val = train_test_split(X.astype(np.float64).values, y.astype(np.float64), test_size=0.2, random_state=42)


def sigmoid(x):
    x = np.array(x, dtype=np.float64)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

#Se definen dos capas ocultas con 32 y 16 neuronas respectivamente
def initialize_weights(input_size, hidden1=32, hidden2=16, output=1):
    np.random.seed(42)
    return (
        np.random.randn(input_size, hidden1).astype(np.float64) * 0.01, np.zeros((1, hidden1), dtype=np.float64),
        np.random.randn(hidden1, hidden2).astype(np.float64) * 0.01, np.zeros((1, hidden2), dtype=np.float64),
        np.random.randn(hidden2, output).astype(np.float64) * 0.01, np.zeros((1, output), dtype=np.float64)
    )

def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)
    return Z1, A1, Z2, A2, Z3, A3

##Try more epochs and less batch size
##Usamos 3000 ejemplos por eficiencia. Se entrenan en bloques de 256 ejemplos
##Por cada batch se calcula forward, el error con cross-entropy y se hace backpropagation hacía el mínimo error
## Se repite el proceso durante 5000 épocas
def train_mini_batch(X, y, X_val, y_val, batch_size=256, epochs=1200, lr=0.05, hidden1=32, hidden2=16):
    np.random.seed(42)
    # Inicialización
    sample_indices = np.random.choice(len(X), size=3000, replace=False)
    X = X[sample_indices].astype(np.float64)
    y = y[sample_indices].astype(np.float64)

    W1, b1, W2, b2, W3, b3 = initialize_weights(X.shape[1], hidden1, hidden2)
    loss_history, train_acc_history, val_acc_history = [], [], []

    # Ciclo de entrenamiento
    for epoch in range(epochs):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        batch_losses = []

        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]

            # Forward pass
            Z1, A1, Z2, A2, Z3, A3 = forward(X_batch, W1, b1, W2, b2, W3, b3)

            # Loss calculation
            loss = -np.mean(y_batch * np.log(A3 + 1e-8) + (1 - y_batch) * np.log(1 - A3 + 1e-8))
            batch_losses.append(loss)

            # Backpropagation
            dZ3 = A3 - y_batch
            dW3 = A2.T @ dZ3 / X_batch.shape[0]
            db3 = np.sum(dZ3, axis=0, keepdims=True) / X_batch.shape[0]

            dA2 = dZ3 @ W3.T
            dZ2 = dA2 * relu_derivative(Z2)
            dW2 = A1.T @ dZ2 / X_batch.shape[0]
            db2 = np.sum(dZ2, axis=0, keepdims=True) / X_batch.shape[0]

            dA1 = dZ2 @ W2.T
            dZ1 = dA1 * relu_derivative(Z1)
            dW1 = X_batch.T @ dZ1 / X_batch.shape[0]
            db1 = np.sum(dZ1, axis=0, keepdims=True) / X_batch.shape[0]

            # Update weights and biases
            W3 -= lr * dW3
            b3 -= lr * db3
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        # Calculate metrics for the epoch
        loss_history.append(np.mean(batch_losses))

        # Training accuracy
        _, _, _, _, _, A3_train = forward(X, W1, b1, W2, b2, W3, b3)
        train_preds = (A3_train > 0.5).astype(int)
        train_acc = accuracy_score(y, train_preds)
        train_acc_history.append(train_acc)

        # Validation accuracy
        _, _, _, _, _, A3_val = forward(X_val, W1, b1, W2, b2, W3, b3)
        val_preds = (A3_val > 0.5).astype(int)
        val_acc = accuracy_score(y_val, val_preds)
        val_acc_history.append(val_acc)

    return W1, b1, W2, b2, W3, b3, loss_history, train_acc_history, val_acc_history

print("Entrenando red neuronal con NumPy...")
start_time = time.time()

W1, b1, W2, b2, W3, b3, losses, train_accuracies, val_accuracies = train_mini_batch(
    X_train, y_train, X_val, y_val
)

train_time = time.time() - start_time
print(f"Tiempo de entrenamiento (NumPy): {train_time:.2f} segundos")

# ================================================
# GRÁFICO DE EVOLUCIÓN DE PRECISIÓN
# ================================================
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy', color='orange')
plt.title('Evolución de Precisión - Entrenamiento vs Validación')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================================
# IMPRIMIR PRECISIONES FINALES
# ================================================
print(f"Precisión final en entrenamiento: {train_accuracies[-1]:.4f}")
print(f"Precisión final en validación: {val_accuracies[-1]:.4f}")