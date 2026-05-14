"""
Visualización de resultados de la simulación.
"""

import numpy as np
import matplotlib.pyplot as plt
from .simulation import run_simulation

ALGORITHM_LABELS = ["Natural", "Anti-Natural", "RBF", "Polinomial"]
METRIC_LABELS    = ["Kernel Alignment", "Kernel Polarization", "FSM", "Complex Ratio"]


def plot_metrics_vs_num_kernels(
    X: np.ndarray,
    y: np.ndarray,
    kernels_list: list,
    n_iterations: int = 10,
    max_features: int = 5,
    power: int = 2,
    figsize: tuple = (12, 8),
) -> None:
    """
    Grafica cómo cambian las métricas al variar el número de kernels.

    Genera una figura 2×2 donde cada subplot muestra una métrica distinta
    en función del número de kernels, para los 4 algoritmos comparados.

    Parámetros
    ----------
    X            : ndarray de datos.
    y            : ndarray de etiquetas.
    kernels_list : lista de enteros — número de kernels a evaluar.
    n_iterations : iteraciones por configuración de kernels.
    max_features : máximo de características por kernel débil.
    power        : exponente de acentuación de pesos.
    figsize      : tamaño de la figura.
    """
    # Ejecutar simulación para cada cantidad de kernels
    results = {
        nk: run_simulation(X, y, n_iterations=n_iterations, num_kernels=nk, max_features=max_features, power=power)
        for nk in kernels_list
    }

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Evolución de Métricas vs Número de Kernels", fontsize=14)

    for i, ax in enumerate(axes.flat):
        for j, label in enumerate(ALGORITHM_LABELS):
            values = [results[nk][j, i] for nk in kernels_list]
            ax.plot(kernels_list, values, marker="o", label=label)
        ax.set_title(METRIC_LABELS[i])
        ax.set_xlabel("Número de Kernels")
        ax.set_ylabel("Valor de Métrica")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
