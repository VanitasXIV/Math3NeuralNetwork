# ================================================
# PARTE 1 - Análisis Exploratorio y Transformaciones
# ================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning  # Importar ConvergenceWarning
import time
import warnings  # Importar módulo de warnings

# Ignorar warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    outliers_report[col] = df[(df[col] < lb) | (df[col] > ub)].shape[0]

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
X = df.drop(columns=['Name', 'Rank', 'Exito_Ventas', 'Year']).astype(np.float64).values
y = df['Exito_Ventas'].values.ravel()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================================
# PARTE 3 - Entrenamiento con Early Stopping y Ajustes
# ================================================
print("Entrenando red neuronal con scikit-learn (MLPClassifier)...")

# Total de iteraciones máximas permitidas
max_epochs = 500  # Incrementar el número máximo de épocas
lr = 0.05  # Se puede reducir la tasa de aprendizaje para mejorar la estabilidad
patience = 20  # Incrementar paciencia para permitir más tiempo de mejora
best_val_acc = 0
epochs_without_improvement = 0

# Configuro el clasificador con ajustes
clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Incrementar la capacidad del modelo
    activation='relu',
    learning_rate_init=lr,
    max_iter=1,           # UNA iteración por llamada a fit()
    warm_start=True,      # conserva los pesos y sigue entrenando
    random_state=42
)

train_accs = []
val_accs = []

start_time = time.time()
for ep in range(1, max_epochs + 1):
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Verificar mejora en validación
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Detener si no hay mejora en 'patience' épocas consecutivas
    if epochs_without_improvement >= patience:
        print(f"Entrenamiento detenido por early stopping en la época {ep}.")
        break

sklearn_train_time = time.time() - start_time

print(f"Última precisión en entrenamiento: {train_accs[-1]:.4f}")
print(f"Última precisión en validación:   {val_accs[-1]:.4f}")
print(f"Tiempo total de entrenamiento:    {sklearn_train_time:.2f} segundos")

# ================================================
# CURVA DE APRENDIZAJE Y VALIDACIÓN
# ================================================
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_accs) + 1), train_accs, label='Entrenamiento')
plt.plot(range(1, len(val_accs) + 1), val_accs, label='Validación')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.title('Curva de Aprendizaje y Validación')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
