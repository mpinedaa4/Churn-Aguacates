"""
Ejemplo de uso completo de la librería extremality_mkl.

Carga el dataset 'australian.csv', prepara las etiquetas y
ejecuta el experimento de comparación de kernels.
"""

import numpy as np
import pandas as pd
import os

from extremality_mkl.validation.plotting import plot_metrics_vs_num_kernels

# ── Carga de datos ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "data", "australian.csv")

data = pd.read_csv(csv_path)
X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

# Convertir etiquetas 0 → -1
y = np.where(y == 0, -1, y)

# ── Parámetros del experimento ───────────────────────────────────────────────
N_ITERATIONS  = 5
MAX_FEATURES  = 5
KERNELS_LIST  = [5, 10, 15]

# ── Ejecutar y graficar ──────────────────────────────────────────────────────
plot_metrics_vs_num_kernels(
    X=X,
    y=y,
    kernels_list=KERNELS_LIST,
    n_iterations=N_ITERATIONS,
    max_features=MAX_FEATURES,
)
