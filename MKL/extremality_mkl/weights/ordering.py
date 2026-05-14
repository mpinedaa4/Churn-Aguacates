"""
Ordenamiento por extremalidad mediante rotación de métricas.

La idea central es: dado un conjunto de kernels, cada uno caracterizado
por varias métricas, ¿cuál es el "mejor" en términos de todas esas
métricas a la vez? En lugar de elegir arbitrariamente una métrica,
se usa una rotación matemática (Gram-Schmidt) para proyectar todas
las métricas en una dirección que maximiza la noción de "extremalidad".
"""

import numpy as np


def _gram_schmidt(A: np.ndarray):
    """
    Ortonormalización de Gram-Schmidt sobre las columnas de A.

    Toma una matriz A de forma (m, n) y devuelve Q (ortonormal)
    y R (triangular superior) tales que A = Q @ R.
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def _rotation_matrix(direction: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de rotación que lleva el vector uniforme (1/√n, …)
    hacia la dirección indicada.

    Se usa para proyectar el espacio de métricas de forma que el
    primer eje apunte en la dirección de interés.

    Parámetros
    ----------
    direction : ndarray de forma (n_metrics,)
        Vector que indica la dirección de "mejora" de cada métrica.

    Retorna
    -------
    ndarray de forma (n_metrics, n_metrics) — matriz de rotación.
    """
    n = len(direction)
    I = np.eye(n)

    # Matriz cuya primera columna es la dirección normalizada
    M_dir = np.sign(direction)[:, None] * I
    M_dir[:, 0] = direction / np.linalg.norm(direction, 2)

    # Matriz cuya primera columna es el vector uniforme
    M_unif = I.copy()
    M_unif[:, 0] = np.ones(n) / np.sqrt(n)

    Q_dir, _ = _gram_schmidt(M_dir)
    Q_unif, _ = _gram_schmidt(M_unif)

    return Q_unif @ Q_dir.T


def extremality_order(metrics_matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Asigna a cada kernel un rango de extremalidad.

    Para cada kernel i, el rango es el número de kernels j tales que
    j domina a i en todas las métricas rotadas (cuanto más alto, más
    kernels lo superan → peor posición).

    Parámetros
    ----------
    metrics_matrix : ndarray de forma (n_kernels, n_metrics)
        Valor de cada métrica para cada kernel.
    direction : ndarray de forma (n_metrics,)
        Signo de cada métrica: +1 si mayor es mejor, -1 si menor es mejor.

    Retorna
    -------
    ndarray de forma (n_kernels,) con el rango de cada kernel (entero ≥ 1).
    """
    R = _rotation_matrix(direction)
    rotated = (R @ metrics_matrix.T).T   # (n_kernels, n_metrics)

    n = rotated.shape[0]
    ranks = np.zeros(n)
    for i in range(n):
        # cuántos kernels dominan a i en todas las métricas rotadas
        ranks[i] = np.sum(np.all(rotated >= rotated[i], axis=1))
    return ranks
