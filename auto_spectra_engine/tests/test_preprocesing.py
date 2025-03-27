import matplotlib.pyplot as plt
import numpy as np
from ..preprocessing import icoshift, baseline_als, apply_phase_correction

def create_complex_spectrum(x, shift=0, baseline_factor=0, phase0=0, phase1=0):
    """
    Create a complex-valued spectrum with simulated baseline and phase distortion.
    """
    # Simulated real peak shape (same as before)
    peak = np.exp(-((x - 5 + shift) ** 2) * 10) + 0.8 * np.exp(-((x - 7 + shift) ** 2) * 20)
    baseline = baseline_factor * (0.5 * x + np.sin(x))
    noise = 0.05 * np.random.randn(len(x))
    real_part = peak + baseline + noise
    imag_part = 0.05 * np.random.randn(len(x))  # imaginary part (just noise here)
    spectrum_complex = real_part + 1j * imag_part

    # Apply phase distortion
    distorted = apply_phase_correction(spectrum_complex, phase0=phase0, phase1=phase1)
    return distorted

# Simulate data
x = np.linspace(0, 10, 500)
shifts = [0, 0.15, -0.25, 0.4]
baseline_levels = [0.3, 0.4, 0.5, 0.6]
phase0s = [45, 20, -30, 60]  # introduce different phase errors
spectra_complex = [
    create_complex_spectrum(x, s, b, p0, 0) for s, b, p0 in zip(shifts, baseline_levels, phase0s)
]
spectra_complex = np.array(spectra_complex)

# -------------------------
# 1. Phase Correction
# -------------------------
corrected_phase = np.array([
    apply_phase_correction(s, phase0=-p)  # correct using inverse of original phase
    for s, p in zip(spectra_complex, phase0s)
])

# -------------------------
# 2. Baseline Correction
# -------------------------
corrected_baseline = np.array([s - baseline_als(s) for s in corrected_phase])

# -------------------------
# 3. Alignment with Icoshift
# -------------------------
ref_spectrum = corrected_baseline[0]
aligned_spectra = icoshift(corrected_baseline, reference=ref_spectrum, intervals=[(0, 500)])

# -------------------------
# 4. Plotting
# -------------------------
plt.figure(figsize=(16, 12))

plt.subplot(4, 1, 1)
plt.title("Original Spectra with Phase and Baseline Distortion")
for s in spectra_complex:
    plt.plot(x, np.real(s))
plt.ylabel("Intensity")

plt.subplot(4, 1, 2)
plt.title("After Phase Correction")
for s in corrected_phase:
    plt.plot(x, s)
plt.ylabel("Intensity")

plt.subplot(4, 1, 3)
plt.title("After Baseline Correction (AsLS)")
for s in corrected_baseline:
    plt.plot(x, s)
plt.ylabel("Intensity")

plt.subplot(4, 1, 4)
plt.title("After Icoshift Alignment")
for s in aligned_spectra:
    plt.plot(x, s)
plt.xlabel("ppm (simulated axis)")
plt.ylabel("Intensity")

plt.tight_layout()
plt.show()