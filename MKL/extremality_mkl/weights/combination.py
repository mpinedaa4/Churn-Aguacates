"""Utilidades para combinar y normalizar pesos."""
import numpy as np


def normalize_weights(ranks: np.ndarray, power: int = 1) -> np.ndarray:
    """
    Convierte un vector de rangos en pesos normalizados que suman 1.

    El proceso es:
      1. Normalizar los rangos para que sumen 1.
      2. Elevar a la potencia `power` para acentuar diferencias.
      3. Renormalizar para que vuelvan a sumar 1.

    Con power=1 los pesos son proporcionales al rango.
    Con power>1 los kernels mejor rankeados reciben peso mucho mayor.

    Parámetros
    ----------
    ranks  : ndarray de forma (n_kernels,)
    power  : int — exponente de acentuación (default 1)

    Retorna
    -------
    ndarray de forma (n_kernels,) con pesos en [0,1] que suman 1.
    """
    w = ranks / ranks.sum()
    w = w ** power
    return w / w.sum()
