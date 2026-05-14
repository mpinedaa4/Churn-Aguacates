"""
Simulación de experimentos para comparar estrategias de combinación de kernels.

Se comparan cuatro estrategias:
  1. Natural     : kernels combinados dando más peso a los de mayor calidad.
  2. Anti-natural: kernels combinados dando más peso a los de menor calidad.
  3. RBF         : un solo kernel RBF global (línea base estándar).
  4. Polinomial  : un solo kernel polinomial de grado 3 (línea base estándar).
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

from ..kernels.polynomial import create_weak_kernels
from ..weights.extremality import kernel_extremality_weights
from ..metrics.kernel_metrics import (
    kernel_alignment,
    kernel_polarization,
    feature_space_measure,
    complex_ratio,
)

METRIC_FUNCTIONS = [kernel_alignment, kernel_polarization, feature_space_measure, complex_ratio]
METRIC_NAMES = ["alignment", "polarization", "FSM", "complex_ratio"]
N_ALGORITHMS = 4  # natural, anti-natural, RBF, polinomial


def _evaluate_kernel(K: np.ndarray, y: np.ndarray) -> list:
    """Evalúa las 4 métricas sobre un kernel dado."""
    return [f(K, y) for f in METRIC_FUNCTIONS]


def _combine_kernels(KL: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combina una lista de kernels usando pesos (suma ponderada)."""
    return np.einsum("ijk,i->jk", KL, weights)


def run_single_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    KL_train: np.ndarray,
    power: int = 2,
) -> np.ndarray:
    """
    Ejecuta un solo experimento y devuelve las métricas de los 4 algoritmos.

    Parámetros
    ----------
    X_train  : ndarray (n_train, n_features)
    y_train  : ndarray (n_train,) con -1 o +1
    KL_train : ndarray (n_kernels, n_train, n_train)
    power    : exponente para normalize_weights

    Retorna
    -------
    ndarray de forma (N_ALGORITHMS, 4) con las métricas de cada algoritmo.
    """
    result = kernel_extremality_weights(KL_train, y_train, power=power)

    K_natural = _combine_kernels(KL_train, result.w_natural)
    K_anti    = _combine_kernels(KL_train, result.w_antinatural)
    K_rbf     = rbf_kernel(X_train)
    K_poly    = polynomial_kernel(X_train, degree=3)

    return np.array([
        _evaluate_kernel(K_natural, y_train),
        _evaluate_kernel(K_anti,    y_train),
        _evaluate_kernel(K_rbf,     y_train),
        _evaluate_kernel(K_poly,    y_train),
    ])


def run_simulation(
    X: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 10,
    num_kernels: int = 10,
    max_features: int = 5,
    test_size: float = 0.3,
    power: int = 2,
) -> np.ndarray:
    """
    Repite el experimento `n_iterations` veces y devuelve las métricas promedio.

    En cada iteración:
      - Se hace un split aleatorio train/test.
      - Se generan `num_kernels` kernels débiles.
      - Se evalúan los 4 algoritmos.

    Parámetros
    ----------
    X            : ndarray (n_samples, n_features) — datos crudos.
    y            : ndarray (n_samples,) con -1 o +1.
    n_iterations : número de repeticiones.
    num_kernels  : cantidad de kernels débiles por iteración.
    max_features : máximo de características por kernel débil.
    test_size    : fracción de datos para test (no se usa en métricas, solo para split).
    power        : exponente de acentuación de pesos.

    Retorna
    -------
    ndarray de forma (N_ALGORITHMS, 4)
        Promedio de cada métrica para cada algoritmo a lo largo de las iteraciones.
    """
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    all_results = np.zeros((n_iterations, N_ALGORITHMS, len(METRIC_FUNCTIONS)))

    for k in range(n_iterations):
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=test_size, random_state=k
        )
        KL_train, _ = create_weak_kernels(
            X_train,
            X_train,  # dummy test para obtener el formato (KL_train, KL_test)
            num_kernels=num_kernels,
            max_features=max_features,
            random_state=k,
        )
        all_results[k] = run_single_experiment(X_train, y_train, KL_train, power)

    return all_results.mean(axis=0)
