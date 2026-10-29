import numpy as np

from src.amplitudes import breit_wigner, flatte, momentum, phase_space, s_matrix_11

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


def test_momentum_sheet_II_is_minus_sheet_I():
    s = np.array([3.85**2, 3.90**2 + 0.01j, 3.88**2 - 0.02j])
    np.testing.assert_allclose(momentum(s, *CH1, sheet=2), -momentum(s, *CH1))


def test_virtual_state_on_sheet_II():
    # Sotto soglia rho_II = -i|rho|: con m^2 = s0 + g^2 |rho(s0)| il
    # denominatore della Flatte' si annulla in s0 sul foglio II, non sul I.
    g, s0 = 0.5, (TH1 - 0.005) ** 2
    rho = abs(phase_space(s0, *CH1))
    m = np.sqrt(s0 + g**2 * rho)
    inv_II = 1 / flatte(s0, m=m, g1=g, g2=0.0, ch1=CH1, sheets=(2, 1))
    inv_I = 1 / flatte(s0, m=m, g1=g, g2=0.0, ch1=CH1, sheets=(1, 1))
    assert abs(inv_II) < 1e-12
    np.testing.assert_allclose(inv_I, 2 * rho)  # sul foglio I: 1/f = 2|rho|


def test_bound_state_on_sheet_I():
    # Caso speculare: m^2 = s0 - g^2 |rho(s0)| da' uno zero sul foglio I.
    g, s0 = 0.5, (TH1 - 0.005) ** 2
    rho = abs(phase_space(s0, *CH1))
    m = np.sqrt(s0 - g**2 * rho)
    inv_I = 1 / flatte(s0, m=m, g1=g, g2=0.0, ch1=CH1, sheets=(1, 1))
    inv_II = 1 / flatte(s0, m=m, g1=g, g2=0.0, ch1=CH1, sheets=(2, 1))
    assert abs(inv_I) < 1e-12
    np.testing.assert_allclose(inv_II, -2 * rho)  # sul foglio II: 1/f = -2|rho|


def test_breit_wigner_peak():
    # In s = m^2: f = i g^2 / (m gamma), fase 90 gradi, |f|^2 massimo.
    m, gamma, g = 3.872, 0.002, 0.5
    f0 = breit_wigner(m**2, m, gamma, g)
    np.testing.assert_allclose(np.degrees(np.angle(f0)), 90.0)
    np.testing.assert_allclose(abs(f0) ** 2, g**4 / (m * gamma) ** 2)
    s = np.linspace(m**2 - 0.1, m**2 + 0.1, 2001)  # griglia centrata in m^2
    assert np.argmax(abs(breit_wigner(s, m, gamma, g))) == 1000


def test_breit_wigner_half_maximum():
    # |f|^2 dimezza in s = m^2 +- m gamma.
    m, gamma, g = 3.872, 0.002, 0.5
    peak = abs(breit_wigner(m**2, m, gamma, g)) ** 2
    for s in (m**2 - m * gamma, m**2 + m * gamma):
        np.testing.assert_allclose(abs(breit_wigner(s, m, gamma, g)) ** 2, peak / 2)
