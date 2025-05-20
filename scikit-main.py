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
from sklearn.neural_network import MLPClassifier
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

# ------------------------------------------------
# Paso 3: Matriz de Correlación
# ------------------------------------------------
numeric_cols = ['Rank', 'Year'] + sales_columns + ['Exito_Ventas']
df_numeric = df[numeric_cols].dropna()
correlation_matrix = df_numeric.corr()
correlation_with_target = correlation_matrix['Exito_Ventas'].sort_values(ascending=False)

# ------------------------------------------------
# Paso 4: Transformaciones Preliminares
# ------------------------------------------------
scaler = MinMaxScaler()
df[sales_columns] = scaler.fit_transform(df[sales_columns])
df = pd.get_dummies(df, columns=['Platform', 'Genre', 'Publisher'], drop_first=True)

# ================================================
# PARTE 2 - Preparación de datos
# ================================================
X = df.drop(columns=['Name', 'Rank', 'Exito_Ventas', 'Year'])
y = df['Exito_Ventas'].values.reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X.astype(np.float64).values, y.astype(np.float64).ravel(), test_size=0.2, random_state=42)

# ================================================
# PARTE 3 - Implementación con scikit-learn
# ================================================
print("Entrenando red neuronal con scikit-learn (MLPClassifier)...")
start_time = time.time()
clf = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='sgd', max_iter=50, random_state=42)
clf.fit(X_train, y_train)
sklearn_train_time = time.time() - start_time

# Evaluación
train_accuracy = clf.score(X_train, y_train)
val_accuracy = clf.score(X_val, y_val)

print(f"Precisión en entrenamiento (sklearn): {train_accuracy:.4f}")
print(f"Precisión en validación (sklearn): {val_accuracy:.4f}")
print(f"Tiempo de entrenamiento (sklearn): {sklearn_train_time:.2f} segundos")


# ================================================
# COMPARACIÓN VISUAL DE RENDIMIENTO
# ================================================
plt.figure(figsize=(8, 5))
bars = plt.bar(['Train Accuracy', 'Validation Accuracy'], [train_accuracy, val_accuracy], color=['blue', 'green'])
plt.ylim(0, 1)
plt.title('Comparación de Precisión - scikit-learn')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
plt.ylabel('Accuracy')
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

