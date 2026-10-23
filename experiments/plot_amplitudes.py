"""Lineshape e fase: Breit-Wigner costante vs Flatte' a due canali."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.amplitudes import breit_wigner, flatte

CH1 = (1.86484, 2.00685)   # D0 D*0
CH2 = (1.86966, 2.01026)   # D+ D*-
TH1, TH2 = sum(CH1), sum(CH2)

E = np.linspace(3.860, 3.900, 4000)
s = E**2

amps = {
    "Breit-Wigner (larghezza costante)": breit_wigner(s, m=3.8716, gamma=0.002),
    "Flatte' a due canali": flatte(s, m=3.8716, g1=0.3, g2=0.3, ch1=CH1, ch2=CH2),
}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
for label, f in amps.items():
    ax1.plot(E, np.abs(f) ** 2 / np.max(np.abs(f) ** 2), label=label)
    ax2.plot(E, np.degrees(np.angle(f)), label=label)
for ax in (ax1, ax2):
    for th in (TH1, TH2):
        ax.axvline(th, color="gray", ls=":", lw=1)
ax1.set_ylabel(r"$|f|^2$ (normalizzato)")
ax2.set_ylabel("fase di f (gradi)")
ax2.set_xlabel(r"$\sqrt{s}$ (GeV)")
ax1.legend()
fig.tight_layout()

out = Path("experiments/figs")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "amplitudes.png", dpi=150)
print("Salvato in", out / "amplitudes.png")
