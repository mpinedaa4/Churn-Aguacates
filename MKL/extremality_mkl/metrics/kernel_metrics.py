"""
Métricas para evaluar la calidad de un kernel.

Todas las funciones reciben una matriz de kernel K (n x n) y,
cuando aplica, el vector de etiquetas y (valores -1 o +1).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Métricas individuales
# ---------------------------------------------------------------------------

def complex_ratio(K: np.ndarray, y: np.ndarray = None) -> float:
    """
    Retorna la traza de la matriz de kernel.

    La traza es la suma de los elementos de la diagonal: equivale a la
    suma de los valores propios, y es una medida de la 'energía' total
    capturada por el kernel.

    Parámetros
    ----------
    K : ndarray de forma (n, n)
        Matriz de kernel simétrica y semidefinida positiva.
    y : ignorado (existe solo para tener la misma firma que las demás métricas).

    Retorna
    -------
    float
        Traza de K.
    """
    return float(np.trace(K))


def ideal_kernel(y: np.ndarray) -> np.ndarray:
    """
    Construye el kernel ideal a partir de las etiquetas de clase.

    El kernel ideal K* tiene K*[i,j] = +1 si y[i]==y[j], y -1 en caso
    contrario. Representa la separación perfecta de clases en el espacio
    de características.

    Parámetros
    ----------
    y : ndarray de forma (n,)
        Etiquetas de clase (-1 o +1).

    Retorna
    -------
    ndarray de forma (n, n)
        Matriz del kernel ideal.
    """
    K_ideal = np.equal.outer(y, y).astype(int)
    K_ideal = np.where(K_ideal == 0, -1, K_ideal)
    return K_ideal


def kernel_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """
    Mide qué tan alineado está el kernel con el kernel ideal (Cristianini, 2002).

    Un valor cercano a 1 indica que el kernel separa muy bien las clases;
    un valor cercano a 0 indica poca discriminación.

    Parámetros
    ----------
    K : ndarray de forma (n, n)
    y : ndarray de forma (n,) con valores -1 o +1

    Retorna
    -------
    float en [0, 1]
    """
    K_ideal = ideal_kernel(y)
    inner_product = np.trace(K.T @ K_ideal)
    norm_K = np.linalg.norm(K, "fro")
    return float(inner_product / (norm_K * len(y)))


def kernel_polarization(K: np.ndarray, y: np.ndarray) -> float:
    """
    Mide la polarización del kernel: qué tanto separa muestras de clases distintas.

    Para cada par (i, j) calcula -y[i]*y[j]*(k[i,i] + k[j,j] - 2*k[i,j]).
    Un valor alto indica buena separación entre clases.

    Parámetros
    ----------
    K : ndarray de forma (n, n)
    y : ndarray de forma (n,) con valores -1 o +1

    Retorna
    -------
    float
    """
    n = len(y)
    # Vectorización: evita el doble for loop del código original
    diag = np.diag(K)                         # (n,)
    diff = diag[:, None] + diag[None, :] - 2 * K  # (n, n) distancias al cuadrado
    label_prod = np.outer(y, y)               # (n, n) producto de etiquetas
    A = -label_prod * diff
    np.fill_diagonal(A, 0)                    # el par (i,i) no aporta
    return float(np.sum(A))


def feature_space_measure(K: np.ndarray, y: np.ndarray) -> float:
    """
    Feature Space Measure (FSM): combina varianza intra-clase e inter-clase.

    Cuantifica la separabilidad de las clases en el espacio inducido por
    el kernel. Un FSM mayor indica mejor capacidad discriminativa.

    Parámetros
    ----------
    K : ndarray de forma (n, n)
    y : ndarray de forma (n,) con valores -1 o +1

    Retorna
    -------
    float
    """
    mask_neg = y == -1
    mask_pos = y == 1
    n_neg = mask_neg.sum()
    n_pos = mask_pos.sum()

    # Similitudes promedio intra e inter clase
    d_i = K[np.ix_(mask_neg, mask_neg)].sum(axis=1) / n_neg  # neg↔neg
    a_i = K[np.ix_(mask_pos, mask_pos)].sum(axis=1) / n_pos  # pos↔pos
    c_i = K[np.ix_(mask_neg, mask_pos)].sum(axis=1) / n_pos  # neg↔pos
    b_i = K[np.ix_(mask_pos, mask_neg)].sum(axis=1) / n_neg  # pos↔neg

    A = a_i.sum() / n_pos
    B = b_i.sum() / n_pos
    C = c_i.sum() / n_neg
    D = d_i.sum() / n_neg

    phi_sq = A + D - B - C  # Factor de normalización

    term_pos = np.sum((b_i - a_i + A - B) ** 2) / (phi_sq * (n_pos - 1))
    term_neg = np.sum((c_i - d_i + D - C) ** 2) / (phi_sq * (n_neg - 1))

    return float((np.sqrt(term_pos) + np.sqrt(term_neg)) / np.sqrt(phi_sq))


# ---------------------------------------------------------------------------
# Registro de métricas disponibles
# ---------------------------------------------------------------------------

METRICS = {
    "alignment":    kernel_alignment,
    "polarization": kernel_polarization,
    "FSM":          feature_space_measure,
    "complex_ratio": complex_ratio,
}

# +1 → mayor es mejor  |  -1 → menor es mejor
DIRECTIONS = {
    "alignment":    1,
    "polarization": 1,
    "FSM":          1,
    "complex_ratio": -1,
}
