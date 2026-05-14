"""
Cálculo de pesos de combinación de kernels por extremalidad.

Se calculan dos conjuntos de pesos:
  - w_natural   : kernels ordenados del mejor al peor (más peso al mejor).
  - w_antinatural: kernels ordenados del peor al mejor (más peso al peor).
                   Se usa como línea base para comparación.
"""

from dataclasses import dataclass
import numpy as np

from ..metrics.kernel_metrics import METRICS, DIRECTIONS, kernel_alignment, feature_space_measure
from .ordering import extremality_order
from .combination import normalize_weights


# Métricas por defecto usadas para el ordenamiento
_DEFAULT_METRICS = {
    "alignment": kernel_alignment,
    "FSM":       feature_space_measure,
}
_DEFAULT_DIRECTIONS = np.array([DIRECTIONS["alignment"], DIRECTIONS["FSM"]])


def _compute_metrics_matrix(
    KL_train: np.ndarray,
    y_train: np.ndarray,
    metrics: dict,
) -> tuple:
    """
    Evalúa cada métrica en cada kernel y devuelve la matriz resultante.

    Retorna
    -------
    metrics_matrix : ndarray (n_kernels, n_metrics)
    directions     : ndarray (n_metrics,)
    """
    n_kernels = KL_train.shape[0]
    metric_names = list(metrics.keys())

    matrix = np.array(
        [[metrics[m](KL_train[i], y_train) for m in metric_names] for i in range(n_kernels)]
    )
    directions = np.array([DIRECTIONS[m] for m in metric_names])
    return matrix, directions


@dataclass
class KernelWeights:
    """
    Contenedor de los dos conjuntos de pesos calculados.

    Atributos
    ---------
    w_natural    : ndarray — pesos que favorecen a los kernels de mayor calidad.
    w_antinatural: ndarray — pesos que favorecen a los kernels de menor calidad.
    """
    w_natural: np.ndarray
    w_antinatural: np.ndarray


def kernel_extremality_weights(
    KL_train: np.ndarray,
    y_train: np.ndarray,
    metrics: dict = None,
    power: int = 1,
) -> KernelWeights:
    """
    Calcula pesos para cada kernel usando un ordenamiento por extremalidad.

    El procedimiento es:
      1. Calcular las métricas seleccionadas para cada kernel.
      2. Rotar el espacio de métricas para encontrar el orden de extremalidad.
      3. Asignar más peso a los kernels con mayor rango (natural) o menor
         rango (antinatural).

    Parámetros
    ----------
    KL_train : ndarray de forma (n_kernels, n, n)
        Conjunto de matrices de kernel de entrenamiento.
    y_train  : ndarray de forma (n,)
        Etiquetas de clase (-1 o +1).
    metrics  : dict, opcional
        Diccionario {nombre: función} de métricas a usar. Por defecto
        usa alignment y FSM.
    power    : int
        Exponente para acentuar diferencias de rango (default 1).

    Retorna
    -------
    KernelWeights
        Objeto con w_natural y w_antinatural.
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS

    metrics_matrix, directions = _compute_metrics_matrix(KL_train, y_train, metrics)
    n_kernels = KL_train.shape[0]

    # Orden natural: más peso a los kernels con mejor rango
    ranks_natural = extremality_order(metrics_matrix, directions)
    w_natural = normalize_weights(n_kernels - ranks_natural + 1, power)

    # Orden antinatural: más peso a los kernels con peor rango
    ranks_anti = extremality_order(metrics_matrix, -directions)
    w_antinatural = normalize_weights(ranks_anti, power)

    return KernelWeights(w_natural=w_natural, w_antinatural=w_antinatural)
