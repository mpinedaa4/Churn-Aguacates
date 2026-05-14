"""
Generación de kernels polinomiales débiles (weak learners).

Un kernel débil se construye eligiendo aleatoriamente un subconjunto
de características del dataset y aplicando un kernel polinomial con
un grado también aleatorio. La idea es similar al bagging en árboles:
muchos modelos simples y diversos se combinan para formar uno robusto.
"""

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel


def create_weak_kernels(
    X_train: np.ndarray,
    X_test: np.ndarray = None,
    num_kernels: int = 10,
    max_features: int = 5,
    max_degree: int = 3,
    random_state: int = None,
) -> tuple:
    """
    Genera un conjunto de kernels polinomiales débiles.

    Por cada kernel se selecciona aleatoriamente:
    - Un subconjunto de columnas (características) del tamaño 1..max_features.
    - Un grado polinomial entre 1 y max_degree.

    Parámetros
    ----------
    X_train : ndarray de forma (n_train, n_features)
        Datos de entrenamiento.
    X_test : ndarray de forma (n_test, n_features), opcional
        Datos de prueba. Si se provee, también se calculan los kernels de prueba.
    num_kernels : int
        Número de kernels débiles a generar.
    max_features : int
        Número máximo de características a seleccionar por kernel.
    max_degree : int
        Grado polinomial máximo.
    random_state : int, opcional
        Semilla para reproducibilidad.

    Retorna
    -------
    KL_train : ndarray de forma (num_kernels, n_train, n_train)
    KL_test  : ndarray de forma (num_kernels, n_test, n_train) — solo si X_test no es None

    Si X_test es None, retorna solo KL_train.
    """
    rng = np.random.default_rng(random_state)

    KL_train = []
    KL_test = [] if X_test is not None else None

    for _ in range(num_kernels):
        n_selected = rng.integers(1, max_features + 1)
        cols = rng.integers(0, X_train.shape[1], size=n_selected)
        degree = int(rng.integers(1, max_degree + 1))

        X_sub = X_train[:, cols]
        KL_train.append(polynomial_kernel(X_sub, degree=degree, coef0=0, gamma=1))

        if X_test is not None:
            X_sub_test = X_test[:, cols]
            KL_test.append(polynomial_kernel(X_sub_test, X_sub, degree=degree, coef0=0, gamma=1))

    KL_train = np.array(KL_train)

    if X_test is not None:
        return KL_train, np.array(KL_test)
    return KL_train
