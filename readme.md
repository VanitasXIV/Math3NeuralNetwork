# Trabajo Práctico Final: Análisis de la Base De Datos

## Objetivo:

Realizar un análisis exhaustivo de una base de datos antes de utilizarla para entrenar una red neuronal de clasificación.

## Selección de la base de Datos:

La base de datos **Video Games Sales Dataset** (https://www.kaggle.com/datasets/zahidmughal2343/video-games-sale) es más adecuada para el Trabajo Práctico, y puede convertirse fácilmente en un problema de clasificación a diferencia de la base de datos de Fórmula 1 (https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) que contiene varias tablas y requiere de varias transformaciones y conexiones entre tablas para poder trabajarla.

## ¿Qué contiene?

Variables como:

1. Nombre del juego

2. Plataforma

3. Año de lanzamiento

4. Género

5. Editorial

6. Ventas en distintas regiones

7. Total global de ventas

##  Ventajas de elegir esta base de datos:

1. Tiene variables numéricas y categóricas.

2. Es fácil de limpiar y transformar.

3. No necesita unir múltiples tablas como la de F1.

4. Puede analizarse con correlaciones, boxplots, codificación y normalización.

5. Perfecto para usar en redes neuronales simples.

##  Plantear un problema de clasificación

## Clasificación de éxito de ventas

Se creará una variable objetivo binaria: “exito\_ventas”

✔️ Permite clasificar si un juego fue **éxito comercial o no**.

## Análisis general de los datos:

1) Descripción de cada columna del conjunto de datos

| Columna | Descripción | Tipo de Variable |
| :---- | :---- | :---- |
| Rank | Posición en el ranking de ventas | Discreta (numérica ordinal) |
| Name | Nombre del videojuego | Categórica (texto) |
| Platform | Consola y sistema en que se lanzó el juego | Categórica  |
| Year | Año de lanzamiento | Discreta (Númerica ordinal) |
| Genre | Género del videojuego | Categórica  |
| Publisher | Empresa que publicó el videojuego | Categórica |
| NA\_Sales | Ventas en Norteamérica | Continua (numérica) |
| EU\_Sales | Ventas en Europa | Continua (numérica) |
| JP\_Sales | Ventas en Japón | Continua (numérica) |
| Other\_Sales | Ventas en otras regiones | Continua (numérica) |
| Global\_Sales | Ventas globales totales | Continua (numérica) |

###  **2\. Análisis de Correlaciones**

Evaluamos qué tan relacionadas están las variables con la columna objetivo **Éxito\_Ventas** (1 si el juego vendió ≥ 1 millón de copias, 0 si no).

#### **Variables más influyentes:**

* Global\_Sales: **0.54** → Correlación directa fuerte (esperado, ya que define la variable objetivo).

* NA\_Sales: **0.50**

* EU\_Sales: **0.49**

* Other\_Sales: **0.42**

* JP\_Sales: **0.33**

Estas ventas regionales tienen **fuerte relación positiva** con el éxito de ventas. Es lógico: un juego que vende mucho en una o varias regiones tiende a superar el millón global.

#### **Variables con menor o inversa relación:**

* Rank: **\-0.57** → Inversamente correlacionado, ya que menor ranking (más alto en la tabla) indica mayor éxito.

* Year: **\-0.11** → Correlación débilmente negativa, posiblemente por tendencias de mercado con el tiempo.

 **Conclusión:**

* Las variables más relevantes para la tarea de clasificación son: **NA\_Sales, EU\_Sales, Other\_Sales, JP\_Sales y Rank**.

* Algunas variables categóricas como Platform, Genre o Publisher podrían ser relevantes también, pero no aparecen en esta matriz porque necesitan codificación para poder analizarse numéricamente.

## **3\. Análisis de Factibilidad**

### **¿Es esta base de datos adecuada para entrenar una red neuronal de clasificación?**

**Sí**, esta base de datos es adecuada para entrenar una red neuronal de clasificación, por las siguientes razones:

* **Columna objetivo clara**: hemos definido una variable binaria Éxito\_Ventas que indica si un juego vendió 1 millón de unidades o más.

* **Suficientes ejemplos**: el dataset contiene miles de registros, lo cual es positivo para que la red neuronal pueda aprender patrones generales sin sobreajustarse fácilmente.

* **Variabilidad en los datos**: incluye tanto juegos exitosos como no exitosos, lo cual es esencial para entrenar un modelo equilibrado.

* **Variables predictoras diversas**:

  * Numéricas: ventas por región, año de lanzamiento, ranking.

  * Categóricas: género, plataforma, editorial.  
     Esto permite entrenar una red con múltiples entradas de distintos tipos.

### **Propósito de entrenar la red**

### El propósito es que la red aprenda a **predecir automáticamente si un nuevo videojuego (con ciertas características conocidas) tendrá éxito comercial**, es decir, si superará el millón de copias vendidas.

Esta predicción puede ayudar a:

* **Estimar el potencial de ventas** de un juego en etapa de diseño.

* **Identificar características comunes** en juegos exitosos.

* **Guiar decisiones de marketing, lanzamiento o inversión.**

### **Objetivo del modelo**

* **Entrada (features):** Plataforma, año, género, editor, ventas regionales, etc.

* **Salida (target):** 0 o 1 → éxito o no en ventas.

* **Tipo de problema:** Clasificación binaria supervisada.

* **Evaluación:** Métricas como accuracy, curva de pérdida, validación cruzada, etc.

### **Observaciones:**

* En todas las variables de ventas (NA\_Sales, EU\_Sales, JP\_Sales, Other\_Sales, Global\_Sales), hay **valores extremadamente altos** (juegos como *Wii Sports* o *Super Mario Bros.* que vendieron decenas de millones).

* Estos puntos aparecen muy separados del resto, lo que confirma que son outliers estadísticamente.

### 

### **¿Deberíamos eliminarlos?**

### **No. Se ha decidido no eliminarlos, y esta es la justificación:**

1. **Los outliers son reales y relevantes.**  
    Representan juegos **extraordinariamente exitosos**, y son precisamente parte del fenómeno que queremos aprender.

2. **No son errores.**  
    No hay evidencia de que se trate de datos incorrectos o mal ingresados.

3. **Enriquecen el modelo.**  
    Ayudan a que la red neuronal aprenda a diferenciar entre juegos comunes y verdaderos superventas.

4. **La normalización posterior mitiga su impacto.**  
    Más adelante vamos a normalizar o escalar los datos, lo que suaviza el efecto de los outliers en el aprendizaje.

## **5\. Transformaciones Preliminares**

##  **1\. Normalización de columnas numéricas**

**Columnas afectadas:**

* NA\_Sales, EU\_Sales, JP\_Sales, Other\_Sales, Global\_Sales

**Técnica utilizada:**

* **MinMaxScaler** → escala todos los valores entre **0 y 1**

**¿Por qué es necesaria?**

* Las redes neuronales son sensibles a la escala de los datos.

* Entrenar con valores muy grandes o desbalanceados hace que el descenso de gradiente sea ineficiente.

* Mejora la estabilidad y velocidad del aprendizaje.

### **2\. Conversión de variables categóricas a numéricas**

**Columnas transformadas:**

* Platform, Genre, Publisher

**Técnica utilizada:**

* **One-Hot Encoding** (con pd.get\_dummies)  
   → genera una columna binaria para cada categoría posible (excepto una, para evitar redundancia)

**¿Por qué es necesaria?**

* Las redes neuronales no entienden texto ni categorías directamente.

* Convertirlas a variables binarias permite que la red aprenda relaciones entre cada categoría y el éxito de ventas.

* One-hot evita asignar valores numéricos arbitrarios que implicarían un orden inexistente.

### **Resultado final**

* El dataset quedó con **16.598 filas** y **627 columnas** (muchas generadas por las variables categóricas).

* Está completamente preparado para ser usado en una red neuronal con NumPy o Scikit-Learn.

## **1\. Arquitectura de la Red Neuronal**

### **Objetivo:**

Clasificar si un videojuego va a ser exitoso en ventas (≥ 1 millón de copias) → problema de **clasificación binaria**.

### **Estructura propuesta:**

* **Entrada:**  
   Número de neuronas \= cantidad de columnas del dataset procesado (por ejemplo, \~600 columnas tras one-hot encoding).  
   *Ejemplo estimado:* 627 columnas

* **Capas ocultas:**

  * Capa 1: 64 neuronas → buena para reducir dimensionalidad y capturar patrones generales.

  * Capa 2: 32 neuronas → permite captar combinaciones más refinadas.

  * Activación: **ReLU** (rápida, evita saturación, buena en redes profundas).

* **Capa de salida:**

  * 1 neurona → salida binaria.

  * Activación: **Sigmoid** → ideal para clasificación binaria, da resultado entre 0 y 1\.

### **Activaciones seleccionadas:**

| Capa | Activación | Justificación |
| :---- | :---- | :---- |
| Ocultas | ReLU | Simple, rápida y evita el problema del gradiente desvanecido. |
| Capa de salida | Sigmoid | Produce probabilidad entre 0 y 1\. Perfecta para clasificación. |

### **Visualización esquemática**

Entrada (X) → \[ReLU\] → Capa oculta 1 (64 neuronas)

             → \[ReLU\] → Capa oculta 2 (32 neuronas)

             → \[Sigmoid\] → Capa de salida (1 neurona)

Pero debido a que no es óptimo manejar semejante cantidad de datos se utilizará un algoritmo por batches y se reducirá la red neuronal a 2 capas de 32 neuronas y 16 neuronas respectivamente

### **Estructura elegida:**

* **Entrada:**  
   Número de neuronas \= cantidad de columnas del dataset procesado (por ejemplo, \~600 columnas tras one-hot encoding).  
   *Ejemplo estimado:* 627 columnas divididas en batches que serán nuestra entrada.

* **Capas ocultas:**

  * Capa 1: 32 neuronas

  * Capa 2: 16 neuronas 

  * Activación: **ReLU** (rápida, evita saturación, buena en redes profundas).

* **Capa de salida:**

  * 1 neurona → salida binaria.

  * Activación: **Sigmoid** → ideal para clasificación binaria, da resultado entre 0 y 1\.

## **Parte 3 \- Comparación de Rendimiento**

## **Comparación NumPy vs scikit-learn (solver='sgd')**

| Aspecto | NumPy (implementación propia) | scikit-learn (MLPClassifier, solver='sgd') |
| :---- | :---- | :---- |
| **Arquitectura** | 2 capas ocultas (32 y 16 neuronas) | Igual |
| **Funciones de activación** | ReLU \+ Sigmoid final | ReLU \+ Logística final (por defecto en clasificación) |
| **Método de entrenamiento** | SGD manual con mini-batch | SGD automático con mini-batch |
| **Tasa de aprendizaje** | 0.05 (fija) | 0.05 (fija) |
| **Épocas** | 50 | 50 |
| **Regularización** | Ninguna | L2 (por defecto, se puede desactivar) |
| **Curva de pérdida** | Implementada manualmente, visible | No accesible directamente (loss\_curve\_ disponible si verbose=True) |
| **Precisión entrenamiento** | \~0.80–0.85 (dependiendo de muestras y ruido) | Similar (\~0.82–0.86) |
| **Precisión validación** | Ligeramente más baja, estable | Similar o ligeramente superior |
| **Tiempo de ejecución** | Mayor (sin optimización vectorial interna) | Mucho menor (con operaciones optimizadas) |
| **Estabilidad y convergencia** | Puede oscilar si no se ajusta correctamente el learning rate | Más estable por validación interna y clipping automático |
| **Curvas de entrenamiento y validación** | Ambas disponibles y comparables gráficamente | Se puede acceder a loss curve, pero no accuracy por epoch directamente |

 **Curvas de entrenamiento y validación**

* **NumPy**:

  * La curva de pérdida decrece gradualmente.

  * La precisión en entrenamiento crece con ruido esperable.

* **scikit-learn**:

  * Comportamiento más suave y estable.

  * No tenemos precisión por época, pero el resultado final es similar o ligeramente mejor.

### **⏱ Tiempo de ejecución**

Se realizaron ejecuciones en dos equipos distintos. En una PC con un CPU Ryzen 7 7800X3D estos fueron los resultados. Esta PC cuenta con una GPU la cual no interviene en el entrenamiento de la red neuronal. A sí mismo la cantidad de System RAM no es un limitante
| Implementación | Tiempo (aprox.) |
| :---- | :---- |
| NumPy | 0.6–0.89 segundos |
| scikit-learn | 0.34-0.46 segundos |

Mientras en una PC con un CPU Ryzen 7 5850U estos fueron los resultados. La cantidad de System RAM no es un limitante
| Implementación | Tiempo (aprox.) |
| :---- | :---- |
| NumPy | 0.7–1.2 segundos |
| scikit-learn | 2.58-5 segundos |

Estimamos que la diferencia tan dispar en ambos equipos se debe a la caché 3D del CPU 7800X3D la cual le permite guardar en caché las instrucciones necesarias para el entrenamiento de la red neuronal y acceder a ellas de forma más inmediata respecto al segundo equipo.

 **Conclusión**

* Ambas redes aprenden correctamente y predicen con precisión similar

* scikit-learn es **más rápida, más robusta, y más adecuada para producción o prototipos rápidos**.

* NumPy es ideal para **aprender el funcionamiento interno**, comprender backpropagation, y experimentar con ajustes personalizados.

Ambas implementaciones son válidas y complementarias según el objetivo en mente: educativo o productivo.

