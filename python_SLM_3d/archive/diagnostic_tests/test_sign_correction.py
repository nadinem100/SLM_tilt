"""
Test if the sign correction actually works by checking a single tweezer at a single z-plane.
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

print("="*70)
print("SIGN CORRECTION TEST")
print("="*70)

# Parameters
wavelength_um = 0.689
focal_length_um = 200000.0  # 200 mm
z_target = 20.0  # Target focus at +20 µm

# SLM parameters (small for speed)
H, W = 512, 512
pixel_um = 3.74

# Create a Gaussian pupil
sigma_px = 100
yy = np.arange(H) - H/2
xx = np.arange(W) - W/2
X, Y = np.meshgrid(xx, yy)
R2_px = X**2 + Y**2
A_in = np.exp(-R2_px / (2 * sigma_px**2))

# Physical coordinates
yy_um = yy * pixel_um
xx_um = xx * pixel_um
X_um, Y_um = np.meshgrid(xx_um, yy_um)
R2_um = X_um**2 + Y_um**2

# Defocus phase correction
k = 2 * np.pi / wavelength_um
f = focal_length_um

print(f"\nSetup:")
print(f"  Focal length: {f/1000:.0f} mm")
print(f"  Target z: {z_target:+.1f} µm")
print(f"  Wavelength: {wavelength_um} µm")

# Test BOTH signs and see which one focuses at the right place
z_scan = np.linspace(-30, 30, 121)

for sign_label, sign in [("POSITIVE (old)", +1), ("NEGATIVE (new)", -1)]:
    print(f"\n--- Testing {sign_label} sign ---")

    # Phase correction to make light focus at z_target
    phase_correction = sign * (k / (2 * f * f)) * z_target * R2_um

    print(f"  Phase correction: RMS = {np.sqrt(np.mean(phase_correction**2)):.6f} rad")
    print(f"  Phase correction: max = {np.max(np.abs(phase_correction)):.6f} rad")

    # Apply correction and propagate
    A_pupil = A_in * np.exp(1j * phase_correction)

    # Scan through z and measure intensity at center
    intensities = []

    for z_um in z_scan:
        # Propagate to this z-plane
        phase_defocus = sign * (k / (2 * f * f)) * z_um * R2_um
        A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
        I = np.abs(A_out)**2
        intensities.append(I[H//2, W//2])

    intensities = np.array(intensities)
    peak_idx = np.argmax(intensities)
    z_peak = z_scan[peak_idx]

    print(f"  Measured peak at: z = {z_peak:+.1f} µm")
    print(f"  Error: {z_peak - z_target:+.1f} µm")

    if abs(z_peak - z_target) < 2.0:
        print(f"  ✓ SUCCESS! Peak is within 2 µm of target")
    else:
        print(f"  ✗ FAIL: Peak is far from target")

print("\n" + "="*70)
print("CONCLUSION:")
print("The sign that produces a peak closest to z_target is the CORRECT sign.")
print("="*70)
