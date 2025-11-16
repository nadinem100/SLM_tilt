"""
Test sign convention EXACTLY as used in the GS algorithm.

The GS algorithm does:
1. Store phi_planes[p] = ±(k / (2f²)) · z_p · R²
2. Forward propagation: A_focal = FFT(A_pupil · exp(+j · phi_planes[p]))
3. Backward propagation: A_pupil' = IFFT(A_focal') · exp(-j · phi_planes[p])

Question: What sign should phi_planes[p] have for positive z_p?
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

print("="*70)
print("GS ALGORITHM SIGN CONVENTION TEST")
print("="*70)

# Setup
wavelength_um = 0.689
focal_length_um = 200000.0
z_target = 20.0  # We want to focus at +20 µm

H, W = 512, 512
pixel_um = 3.74
sigma_px = 80

yy = np.arange(H) - H/2
xx = np.arange(W) - W/2
X, Y = np.meshgrid(xx, yy)
R2_px = X**2 + Y**2
A_in = np.exp(-R2_px / (2 * sigma_px**2))

X_um = X * pixel_um
Y_um = Y * pixel_um
R2_um = X_um**2 + Y_um**2

k = 2 * np.pi / wavelength_um
f = focal_length_um

print(f"\nSetup:")
print(f"  Target z: {z_target:+.1f} µm")
print(f"  Focal length: {f/1000:.0f} mm")

# Test both sign conventions
for sign_name, sign in [("NEGATIVE (current)", -1), ("POSITIVE (original)", +1)]:
    print(f"\n--- Testing {sign_name}: phi = {sign:+d} · (k/(2f²)) · z · R² ---")

    # Store the phase correction as done in the class
    phi_plane = sign * (k / (2 * f * f)) * z_target * R2_um

    print(f"  phi_plane RMS: {np.sqrt(np.mean(phi_plane**2)):.6f} rad")

    # Forward propagation (as in GS algorithm line 885)
    A_focal = fftshift(fft2(ifftshift(A_in * np.exp(1j * phi_plane))))

    # This should give us the field at the plane z = z_target
    # Check: where is the peak intensity?
    I_focal = np.abs(A_focal)**2
    center_intensity = I_focal[H//2, W//2]

    # Now propagate to different z-planes and see where peak is
    # We'll mimic what the visualization code does

    z_scan = np.linspace(-30, 30, 121)
    intensities = []

    for z_test in z_scan:
        # Apply defocus propagation (as in visualization code)
        # With NEW sign convention:
        phase_defocus = -sign * (k / (2 * f * f)) * z_test * R2_um
        A_out = fftshift(fft2(ifftshift(A_in * np.exp(1j * (phi_plane + phase_defocus)))))
        I = np.abs(A_out)**2
        intensities.append(I[H//2, W//2])

    intensities = np.array(intensities)
    peak_idx = np.argmax(intensities)
    z_peak = z_scan[peak_idx]

    print(f"  Measured peak at z = {z_peak:+.1f} µm")
    print(f"  Error: {z_peak - z_target:+.1f} µm")

    if abs(z_peak - z_target) < 2.0:
        print(f"  ✓ CORRECT!")
    else:
        print(f"  ✗ WRONG")

print("\n" + "="*70)
print("INTERPRETATION:")
print("The sign convention that gives z_peak ≈ z_target is CORRECT.")
print("="*70)
