"""Ampiezze di scattering per lineshape vicino a soglia.

Convenzioni
-----------
- Variabile: s in GeV^2, complessa in generale.
- Fase spazio a due corpi: rho(s) = 2 q(s) / sqrt(s).
- Ampiezza a un canale normalizzata in modo che S = 1 + 2 i rho f,
  quindi l'unitarieta' elastica e' Im(1/f) = -rho sopra soglia.
"""
import numpy as np


def momentum(s, m1, m2, sheet=1):
    """Impulso nel centro di massa q(s), continuato nel piano complesso.

    Foglio I: Im q >= 0 (sopra soglia, sull'asse reale, q > 0).
    Foglio II: q_II(s) = -q_I(s).
    """
    s = np.asarray(s, dtype=complex)
    lam = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)  # funzione di Kallen
    q = np.sqrt(lam) / (2 * np.sqrt(s))
    q = np.where(q.imag < 0, -q, q)                      # forza Im q >= 0
    q = np.where((q.imag == 0) & (q.real < 0), -q, q)   # sull'asse: q > 0
    return q if sheet == 1 else -q


def phase_space(s, m1, m2, sheet=1):
    """rho(s) = 2 q(s) / sqrt(s)."""
    s = np.asarray(s, dtype=complex)
    return 2 * momentum(s, m1, m2, sheet) / np.sqrt(s)


def breit_wigner(s, m, gamma, g=1.0):
    """Breit-Wigner a larghezza costante. Non unitaria: solo per confronto."""
    s = np.asarray(s, dtype=complex)
    return g**2 / (m**2 - s - 1j * m * gamma)


def flatte(s, m, g1, g2, ch1, ch2=None, sheets=(1, 1)):
    """Flatte' a uno o due canali; restituisce l'ampiezza del canale 1.

    ch1, ch2 : tuple (m_a, m_b) con le masse dei due mesoni di ciascun canale.
    sheets   : foglio (1 o 2) di ciascun canale, per la continuazione analitica.
    Con g2 = 0 (o ch2 = None) si riduce a una Breit-Wigner unitaria
    con larghezza dipendente dall'energia.
    """
    s = np.asarray(s, dtype=complex)
    denom = m**2 - s - 1j * g1**2 * phase_space(s, *ch1, sheet=sheets[0])
    if ch2 is not None:
        denom = denom - 1j * g2**2 * phase_space(s, *ch2, sheet=sheets[1])
    return g1**2 / denom


def s_matrix_11(s, f, ch1):
    """Elemento S_11 = 1 + 2 i rho_1 f (sul foglio fisico)."""
    return 1 + 2j * phase_space(s, *ch1) * f
