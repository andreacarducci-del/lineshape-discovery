import numpy as np

from src.amplitudes import flatte, momentum, s_matrix_11

# Masse in GeV: D0, D*0, D+, D*-
CH1 = (1.86484, 2.00685)
CH2 = (1.86966, 2.01026)
TH1, TH2 = sum(CH1), sum(CH2)


def test_momentum_zero_at_threshold():
    assert abs(momentum(TH1**2, *CH1)) < 1e-6


def test_momentum_imaginary_below_threshold():
    q = momentum((TH1 - 0.01) ** 2, *CH1)
    assert abs(q.real) < 1e-12 and q.imag > 0


def test_unitarity_single_channel():
    s = np.linspace(TH1 + 1e-4, TH1 + 0.1, 200) ** 2
    f = flatte(s, m=3.872, g1=0.5, g2=0.0, ch1=CH1)
    np.testing.assert_allclose(np.abs(s_matrix_11(s, f, CH1)), 1.0, atol=1e-10)


def test_unitarity_between_thresholds():
    s = np.linspace(TH1 + 1e-5, TH2 - 1e-5, 200) ** 2
    f = flatte(s, m=3.872, g1=0.5, g2=0.5, ch1=CH1, ch2=CH2)
    np.testing.assert_allclose(np.abs(s_matrix_11(s, f, CH1)), 1.0, atol=1e-10)


def test_inelasticity_above_second_threshold():
    s = np.linspace(TH2 + 1e-3, TH2 + 0.05, 200) ** 2
    f = flatte(s, m=3.872, g1=0.5, g2=0.5, ch1=CH1, ch2=CH2)
    assert np.all(np.abs(s_matrix_11(s, f, CH1)) < 1)
