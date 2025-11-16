"""
Test Fourier propagation to understand sign conventions.

The key insight: In Fourier optics, to focus at z > 0 (beyond focal plane),
we need a DIVERGING wavefront, which requires POSITIVE quadratic phase.

This is because:
- Converging lens brings light to focus at f
- To push focus FARTHER (z > 0), we need to make it LESS converging (add diverging phase)
- Diverging phase = POSITIVE R^2 term

Reference: Goodman, "Introduction to Fourier Optics", Chapter 5
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

print("="*70)
print("FOURIER PROPAGATION SIGN TEST")
print("="*70)

# Simple test: Apply a lens phase and see where it focuses
wavelength_um = 0.689
focal_length_um = 200000.0

# Pupil
H, W = 512, 512
pixel_um = 3.74
sigma_px = 80

yy = np.arange(H) - H/2
xx = np.arange(W) - W/2
X, Y = np.meshgrid(xx, yy)
R2_px = X**2 + Y**2
A_in = np.exp(-R2_px / (2 * sigma_px**2))

# Physical coordinates
X_um = X * pixel_um
Y_um = Y * pixel_um
R2_um = X_um**2 + Y_um**2

k = 2 * np.pi / wavelength_um
f = focal_length_um

# Test: Apply a "push focus farther by delta_z" correction
delta_z = 20.0  # Want to focus 20 um BEYOND the nominal focal plane

print(f"\nGoal: Focus at z = +{delta_z} µm (beyond focal plane)")
print(f"Focal length: {f/1000:.0f} mm")

# According to thin lens formula, to add defocus delta_z:
# Phase at pupil = (k / 2f^2) * delta_z * R^2
# Question: What SIGN?

# Physical reasoning:
# - Converging lens (negative phase) brings light to focus at z=0 (f)
# - To move focus to z > 0 (farther), we need LESS convergence
# - Less convergence = add POSITIVE (diverging) phase
# - So: phi = +k/(2f^2) * delta_z * R^2 for delta_z > 0

for sign_name, phase_sign in [("POSITIVE (diverging)", +1), ("NEGATIVE (converging)", -1)]:
    print(f"\n--- {sign_name} phase at pupil ---")

    # Phase to apply at pupil
    phi_pupil = phase_sign * (k / (2 * f * f)) * delta_z * R2_um
    print(f"  Applied phase: sign={phase_sign:+d}, RMS={np.sqrt(np.mean(phi_pupil**2)):.6f} rad")

    # Field at pupil
    U_pupil = A_in * np.exp(1j * phi_pupil)

    # Propagate with Fourier transform (this gives us the focal plane at f)
    U_focal_at_f = fftshift(fft2(ifftshift(U_pupil)))
    I_at_f = np.abs(U_focal_at_f)**2

    # But we want to know: does this focus at z=0 or z=delta_z?
    # To check, we propagate to various z-planes using Fresnel propagation

    z_scan = np.linspace(-30, 30, 121)
    intensities = []

    for z_test in z_scan:
        # Propagate from pupil to z = f + z_test
        # In the Fourier domain (after lens), this is a simple phase multiplication
        phase_prop = (k / (2 * f * f)) * z_test * R2_um
        U_test = A_in * np.exp(1j * (phi_pupil + phase_prop))
        U_focal = fftshift(fft2(ifftshift(U_test)))
        I = np.abs(U_focal)**2
        intensities.append(I[H//2, W//2])

    intensities = np.array(intensities)
    peak_idx = np.argmax(intensities)
    z_peak = z_scan[peak_idx]

    print(f"  Measured peak at z = {z_peak:+.1f} µm")
    print(f"  Error from target: {z_peak - delta_z:+.1f} µm")

    if abs(z_peak - delta_z) < 2.0:
        print(f"  ✓ CORRECT: Peak matches target!")
    else:
        print(f"  ✗ WRONG: Peak far from target")

print("\n" + "="*70)
