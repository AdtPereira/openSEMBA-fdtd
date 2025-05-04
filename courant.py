import numpy as np
from scipy.constants import c, mu_0, epsilon_0

def courant_dt(dx, dy, dz, alfa=0, epsilon_r=1.0, mu_r=1.0):
    """
    Calcula o passo de tempo máximo (Delta t) para FDTD 3D baseado na condição de Courant.

    Parameters
    ----------
    dx : float
        Passo espacial na direção x (em metros).
    dy : float
        Passo espacial na direção y (em metros).
    dz : float
        Passo espacial na direção z (em metros).
    epsilon_r : float, optional
        Permissividade relativa do material (default é 1.0 — vácuo).
    mu_r : float, optional
        Permeabilidade relativa do material (default é 1.0 — vácuo).

    Returns
    -------
    delta_t : float
        Passo de tempo máximo (em segundos).
    """

    # Velocidade da luz no meio
    v = c / np.sqrt(epsilon_r * mu_r)

    # Condição de Courant 3D
    inv_sum = (1 / dx**2) + (1 / dy**2) + (1 / dz**2)
    return 1 / (v * np.sqrt(inv_sum)) * (1 - alfa)

# Exemplo de uso
delta_t = courant_dt(dx=0.100, dy=0.100, dz=0.100, alfa=0, epsilon_r=1, mu_r=1)
print(f"Passo de tempo (delta_t): {delta_t:.6e} s")
