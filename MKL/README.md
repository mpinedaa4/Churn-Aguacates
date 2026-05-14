# extremality_mkl

Librería Python para **combinación de múltiples kernels (MKL)** usando un ordenamiento por extremalidad. En lugar de elegir a mano un único kernel para un modelo, esta librería genera muchos kernels débiles, los evalúa con métricas de calidad y los combina automáticamente dándole más peso a los mejores.

---

## ¿Qué problema resuelve?

En modelos como SVM, la elección del kernel (RBF, polinomial, lineal) afecta drásticamente el rendimiento. Normalmente esta elección se hace por prueba y error. `extremality_mkl` propone una alternativa:

1. **Generar muchos kernels débiles** (cada uno usa una combinación aleatoria de características con un grado polinomial aleatorio).
2. **Medir la calidad de cada kernel** con 4 métricas complementarias.
3. **Ordenarlos por extremalidad**: un método matemático que considera *todas* las métricas a la vez (sin elegir arbitrariamente cuál importa más).
4. **Combinarlos** en un único kernel ponderado, listo para usar en cualquier modelo.

---

## Estructura del proyecto

```
MKL/                        <- carpeta raíz
├── extremality_mkl/                <- la librería
│   ├── __init__.py
│   ├── metrics/
│   │   └── kernel_metrics.py      <- 4 métricas de calidad de kernels
│   ├── kernels/
│   │   └── polynomial.py          <- generación de kernels débiles
│   ├── weights/
│   │   ├── ordering.py            <- ordenamiento por extremalidad
│   │   ├── combination.py         <- normalización de pesos
│   │   └── extremality.py         <- función principal de pesos
│   └── validation/
│       ├── simulation.py          <- experimento de comparación
│       ├── plotting.py            <- visualización de resultados
│       └── example.py             <- script de ejemplo
├── data/
│   └── mi_dataset.csv             <- el dataset aquí
├── requirements.txt
├── setup.py
└── test.py                        <- script de prueba
```

> **Importante:** la carpeta `extremality_mkl/` y el archivo `setup.py` deben estar en la **raíz del proyecto**, al mismo nivel que `test.py` y `requirements.txt`.

---

## Instalación

### Requisitos previos

- Python 3.8 o superior
- pip

### Pasos

**1. Crear un entorno virtual**

```bash
python -m venv venv

# En Linux / macOS:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

**2. Instalar las dependencias**

```bash
pip install -r requirements.txt
```

**3. Instalar la librería en modo editable**

```bash
pip install -e .
```

El modo editable (`-e`) permite que los cambios que hagas en el código fuente se reflejen de inmediato sin reinstalar.

---

## Dependencias

| Paquete | Para qué se usa |
|---|---|
| `numpy` | Operaciones matriciales y álgebra lineal |
| `scikit-learn` | Kernels base (polinomial, RBF) y preprocesamiento |
| `matplotlib` | Gráficas de resultados |
| `pandas` | Carga de datasets en CSV |

---

## Uso rápido

### Caso 1 — Solo calcular pesos para los kernels

Si ya se tiene matrices de kernel o los datos listos:

```python
import numpy as np
from extremality_mkl import kernel_extremality_weights, create_weak_kernels

# Suponiendo que se tiene X_train (n_muestras, n_features) e y_train (n_muestras,)
# Las etiquetas deben ser -1 o +1 (no 0 y 1)

# 1. Generar kernels débiles
KL_train = create_weak_kernels(
    X_train,
    num_kernels=10,     # cuántos kernels generar
    max_features=5,     # máximo de características por kernel
    max_degree=3,       # grado polinomial máximo
    random_state=42,    # para reproducibilidad
)
# KL_train tiene forma (10, n_muestras, n_muestras)

# 2. Calcular pesos por extremalidad
weights = kernel_extremality_weights(KL_train, y_train)

# 3. Combinar los kernels en uno solo
K_combined = np.einsum("ijk,i->jk", KL_train, weights.w_natural)
# K_combined es tu nuevo kernel (n_muestras, n_muestras)

# 4. Usar el kernel combinado en un SVM (o cualquier modelo)
from sklearn.svm import SVC
model = SVC(kernel="precomputed")
model.fit(K_combined, y_train)
```

### Caso 2 — Comparar estrategias y ver gráficas

```python
import numpy as np
from extremality_mkl.validation.plotting import plot_metrics_vs_num_kernels

# X: matriz de features, y: etiquetas en -1/+1
plot_metrics_vs_num_kernels(
    X=X,
    y=y,
    kernels_list=[5, 10, 15, 20],   # números de kernels a comparar
    n_iterations=10,                 # repeticiones por configuración
    max_features=5,
    power=2,                         # exponente de acentuación de pesos
)
```

Esto genera una figura 2×2 comparando 4 algoritmos (Natural, Anti-Natural, RBF, Polinomial) sobre 4 métricas, para cada número de kernels indicado.

---

## Usar con tu propio dataset CSV

### Formato esperado del CSV

El CSV debe tener:
- **Todas las columnas excepto la última**: características (features).
- **Última columna**: etiqueta de clase (debe ser binaria: `0` y `1`, o `-1` y `1`).

Ejemplo de CSV válido (`mi_dataset.csv`):

```
feature_1,feature_2,feature_3,clase
0.5,1.2,3.1,0
1.3,0.8,2.7,1
0.9,1.5,3.4,0
...
```

### Script completo para tu CSV

```python
import numpy as np
import pandas as pd
from extremality_mkl.validation.plotting import plot_metrics_vs_num_kernels

# ── 1. Cargar el dataset ──────────────────────────────────────────
data = pd.read_csv("ruta/a/mi_dataset.csv")

# Separar features (X) y etiquetas (y)
X = data.iloc[:, :-1].to_numpy()   # todas las columnas menos la última
y = data.iloc[:, -1].to_numpy()    # última columna

# ── 2. Convertir etiquetas a formato -1 / +1 ─────────────────────
# Si tus etiquetas son 0 y 1:
y = np.where(y == 0, -1, y)
# Si ya son -1 y +1, omite esta línea.

# ── 3. Ejecutar el experimento y graficar ─────────────────────────
plot_metrics_vs_num_kernels(
    X=X,
    y=y,
    kernels_list=[5, 10, 15],   # prueba con distintas cantidades de kernels
    n_iterations=5,              # aumenta para resultados más estables
    max_features=5,              # no debe superar el número de columnas de X
)
```

### Si solo se quiere el kernel combinado (sin gráficas)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from extremality_mkl import kernel_extremality_weights, create_weak_kernels

# 1. Cargar y preparar datos
data = pd.read_csv("ruta/a/mi_dataset.csv")
X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()
y = np.where(y == 0, -1, y)   # convertir etiquetas si es necesario

# 2. Normalizar y dividir
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Generar kernels débiles (para train Y test)
KL_train, KL_test = create_weak_kernels(
    X_train, X_test,
    num_kernels=15,
    max_features=5,
    random_state=42,
)

# 4. Calcular pesos y combinar kernels
weights = kernel_extremality_weights(KL_train, y_train, power=2)
K_train = np.einsum("ijk,i->jk", KL_train, weights.w_natural)
K_test  = np.einsum("ijk,i->jk", KL_test,  weights.w_natural)

# 5. Entrenar y evaluar SVM
model = SVC(kernel="precomputed")
model.fit(K_train, y_train)
y_pred = model.predict(K_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## Referencia de las métricas

Estas son las 4 métricas que la librería usa para evaluar cada kernel:

| Métrica | Función | Interpretación |
|---|---|---|
| **Kernel Alignment** | `kernel_alignment(K, y)` | Qué tan alineado está el kernel con la separación perfecta de clases. Rango ≈ [0, 1], mayor es mejor. |
| **Kernel Polarization** | `kernel_polarization(K, y)` | Qué tanto empuja hacia lados opuestos a ejemplos de distintas clases. Mayor es mejor. |
| **FSM** | `feature_space_measure(K, y)` | Combina varianza intra-clase e inter-clase. Mayor indica mejor separabilidad. |
| **Complex Ratio** | `complex_ratio(K)` | Traza de la matriz (energía total). En este contexto, menor es mejor. |

Se pueden usar de forma independiente:

```python
from extremality_mkl.metrics import kernel_alignment, feature_space_measure

score = kernel_alignment(K, y)
print(f"Alignment: {score:.4f}")
```

---

## Parámetros principales

### `create_weak_kernels`

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `X_train` | ndarray | — | Datos de entrenamiento |
| `X_test` | ndarray | `None` | Datos de prueba (opcional) |
| `num_kernels` | int | `10` | Número de kernels a generar |
| `max_features` | int | `5` | Máximo de features por kernel |
| `max_degree` | int | `3` | Grado polinomial máximo |
| `random_state` | int | `None` | Semilla para reproducibilidad |

### `kernel_extremality_weights`

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `KL_train` | ndarray | — | Conjunto de matrices de kernel `(n_kernels, n, n)` |
| `y_train` | ndarray | — | Etiquetas de clase (`-1` o `+1`) |
| `metrics` | dict | `alignment + FSM` | Métricas a usar para el ordenamiento |
| `power` | int | `1` | Exponente de acentuación: valores altos concentran más peso en los mejores kernels |

El objeto retornado `KernelWeights` tiene dos atributos:
- `w_natural`: pesos que favorecen a los kernels de mayor calidad.
- `w_antinatural`: pesos que favorecen a los de menor calidad (útil como línea base negativa).

---

## Ejemplo incluido

El archivo `extremality_mkl/validation/example.py` contiene un ejemplo completo listo para correr con el dataset `australian.csv`. Para ejecutarlo:

```bash
python -m extremality_mkl.validation.example
```

Asegúrese de que el CSV esté en la carpeta `data/` en la raíz del proyecto.

---

## Probar la librería con `test.py`

El archivo `test.py` (en la raíz del proyecto) ejecuta un flujo completo: carga tu CSV, genera kernels débiles, calcula pesos por extremalidad, entrena 4 SVMs y compara sus resultados.

### Pasos para ejecutarlo

**1. Asegúrese de tener la estructura correcta** (ver sección anterior).

**2. Poner el CSV** en la carpeta `data/` o ajustar la ruta en `test.py`:

```python
CSV_PATH = "data/mi_dataset.csv"   # <- cambiar esto si es necesario
```

**3. Activa tu entorno virtual**:

```bash
# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**4. Instalar las dependencias** (solo la primera vez):

```bash
pip install -r requirements.txt
pip install -e .
```

**5. Correr el script desde la raíz del proyecto:**

```bash
python test.py
```

### Qué hace `test.py`

| Paso | Qué ocurre |
|---|---|
| 1. Carga datos | Lee el CSV, separa features y etiquetas, convierte 0→-1 si hace falta |
| 2. Preprocesa | Normaliza con MinMaxScaler, divide en train/test (70/30) |
| 3. Kernels débiles | Genera `NUM_KERNELS` kernels polinomiales aleatorios |
| 4. Pesos | Calcula `w_natural` y `w_antinatural` por extremalidad |
| 5. Comparación | Entrena 4 SVMs: MKL Natural, MKL Anti-Natural, RBF, Polinomial |
| 6. Resultados | Imprime accuracy y reporte detallado del mejor modelo |
| 7. Gráficas | Opcional: muestra métricas vs número de kernels |

### Salida esperada

```
============================================================
EXTREMALITY MKL — Test completo
============================================================

[1/5] Cargando dataset: data/australian.csv
      Forma del dataset: (690, 15) (690 muestras, 14 features)
      Clases: [-1  1] | Distribución: {-1: 383, 1: 307}

[2/5] Preprocesando y dividiendo datos...
      Train: 483 muestras | Test: 207 muestras

[3/5] Generando 15 kernels débiles (max_features=5)...
      KL_train: (15, 483, 483) | KL_test: (15, 207, 483)

[4/5] Calculando pesos por extremalidad (power=2)...
      Top 5 kernels por peso natural:
        #1  kernel 07 → peso 0.1823
        ...

[5/5] Entrenando SVMs con distintos kernels...

============================================================
RESULTADOS — Accuracy en test
============================================================
  MKL Natural            0.8647  ██████████████████████████
  MKL Anti-Natural       0.7971  ████████████████████████
  RBF (baseline)         0.8551  █████████████████████████
  Poly-3 (baseline)      0.8213  ████████████████████████

  ✓ Mejor estrategia: MKL Natural (0.8647)
```

### Ajustar parámetros

Al inicio de `test.py` puedes modificar:

```python
CSV_PATH     = "data/mi_dataset.csv"  # ruta al CSV
NUM_KERNELS  = 15    # más kernels = más estable, más lento
MAX_FEATURES = 5     # no debe superar el número de columnas del CSV
POWER        = 2     # mayor = más peso al mejor kernel
TEST_SIZE    = 0.3   # fracción de datos para test
```

---

## Preguntas frecuentes

**¿Qué pasa si mis etiquetas no son -1 y +1?**
Conviértalas antes de usar la librería: `y = np.where(y == 0, -1, y)`.

**¿Cuántos kernels debo generar?**
Empezar con 10–20. Más kernels dan resultados más estables pero toman más tiempo. Usar `kernels_list=[5, 10, 20]` en `plot_metrics_vs_num_kernels` para ver el efecto en tus datos.

**¿Funciona con más de 2 clases?**
No. La versión actual está diseñada para clasificación binaria (2 clases: -1 y +1).

**¿Puedo usar mis propias métricas?**
Sí. Cualquier función con firma `f(K: ndarray, y: ndarray) -> float` puede pasarse en el parámetro `metrics` de `kernel_extremality_weights`. Recordar agregar su dirección (`+1` o `-1`) al diccionario `DIRECTIONS` en `metrics/kernel_metrics.py`.