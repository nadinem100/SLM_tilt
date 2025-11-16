"""
Visualize individual tweezer focal positions to verify tilt.
This plots intensity vs z for each tweezer to find where it focuses.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers

print("="*70)
print("INDIVIDUAL TWEEZER FOCUS POSITIONS")
print("="*70)

# Small test
N_HORIZ = 5
N_VERT = 1  # Just one row for clarity
SPACING_UM = 100.0  # Larger spacing
FOCAL_LENGTH_UM = 200000.0  # Test with 200mm like the main experiment
TILT_ANGLE_X = 5  # 5 degrees

# Setup
slm = SLMTweezers(yaml_path="../slm_parameters.yml", redSLM=1, scal=4)
slm.init_fields(waist_um=4500)
slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM, odd_tw=1, box1=2)
slm.set_optics(wavelength_um=0.689, focal_length_um=FOCAL_LENGTH_UM)
slm.assign_planes_from_tilt(tilt_x_deg=TILT_ANGLE_X, n_planes=5)

print(f"\n--- Running GS (10 iterations) ---")
slm.run_gs_multiplane_v3(iterations=10, Gg=0.6, verbose=False, tol=1e-5)

# Propagate and measure intensity at each tweezer position
phase = slm.phase_mask
A_in = slm.A_in
pixel_um = slm.params.pixel_um

print(f"\n--- Measuring focal positions ---")

# Build phase field
H, W = A_in.shape
h, w = phase.shape
psi_full = np.zeros((H, W), dtype=np.float32)
y0 = (H - h) // 2
x0 = (W - w) // 2
psi_full[y0:y0 + h, x0:x0 + w] = phase

# Pupil field
A_pupil = A_in * np.exp(1j * psi_full)

# Pupil coordinates
yy = (np.arange(H) - H / 2) * pixel_um
xx = (np.arange(W) - W / 2) * pixel_um
X, Y = np.meshgrid(xx, yy)
R2 = X**2 + Y**2

# Focal plane coordinates
wavelength_um = 0.689
k = 2 * np.pi / wavelength_um
f = FOCAL_LENGTH_UM
px_focal_um = (wavelength_um * f) / (W * pixel_um)
x_focal = (np.arange(W) - W / 2) * px_focal_um
y_focal = (np.arange(H) - H / 2) * px_focal_um

# Z range to scan - VERY EXPANDED to see full intensity profile and peak
z_min = -1000
z_max = +1000
z_array = np.linspace(z_min, z_max, 401)

# Expected tweezer positions (x, z)
x_tweezers = slm.target_xy_um[:, 0]
z_expected = slm._z_per_spot

print(f"  Expected tweezer positions (x, z):")
for i in range(len(x_tweezers)):
    print(f"    Tweezer {i}: x={x_tweezers[i]:+.1f} µm, z={z_expected[i]:+.1f} µm")

# For each tweezer, find where in z it focuses
focal_positions = []
intensity_curves = []

for tw_idx in range(len(x_tweezers)):
    x_tw = x_tweezers[tw_idx]

    # Find focal plane pixel closest to this x position
    x_px_idx = np.argmin(np.abs(x_focal - x_tw))

    # Scan through z and measure intensity at this (x, y=center) position
    intensities = []

    for z_um in z_array:
        # Paraxial propagation
        phase_defocus = +(k / (2 * f * f)) * z_um * R2
        A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
        I = np.abs(A_out)**2

        # Measure intensity at tweezer position (center row, specific x)
        intensity_at_tw = I[H//2, x_px_idx]
        intensities.append(intensity_at_tw)

    intensities = np.array(intensities)
    intensity_curves.append(intensities)

    # Find peak
    peak_idx = np.argmax(intensities)
    z_peak = z_array[peak_idx]
    focal_positions.append(z_peak)

    print(f"    Tweezer {tw_idx} focuses at z={z_peak:+.1f} µm (expected {z_expected[tw_idx]:+.1f} µm)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Intensity vs z for each tweezer
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(x_tweezers)))
for i, (intensities, x_tw, z_exp) in enumerate(zip(intensity_curves, x_tweezers, z_expected)):
    norm_int = intensities / intensities.max()
    ax.plot(z_array, norm_int, color=colors[i], linewidth=2,
            label=f'x={x_tw:+.0f}µm (expect z={z_exp:+.1f}µm)')
    # Mark expected position
    ax.axvline(z_exp, color=colors[i], linestyle='--', alpha=0.5, linewidth=1)

ax.set_xlabel('z [µm]', fontsize=13)
ax.set_ylabel('Normalized Intensity', fontsize=13)
ax.set_title('Intensity vs z for each tweezer', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='upper right')

# Right: Measured vs expected focal positions
ax = axes[1]
ax.scatter(x_tweezers, focal_positions, s=100, c='blue', marker='o',
           label='Measured peak', zorder=3, edgecolors='black', linewidths=2)
ax.scatter(x_tweezers, z_expected, s=100, c='red', marker='x',
           label='Expected (from algorithm)', zorder=3, linewidths=2)

# Ideal tilt line
x_line = np.array([x_tweezers.min(), x_tweezers.max()])
z_line = np.tan(np.deg2rad(TILT_ANGLE_X)) * x_line
ax.plot(x_line, z_line, 'k--', linewidth=2, alpha=0.7, label=f'Ideal tilt ({TILT_ANGLE_X}°)')

ax.set_xlabel('x position [µm]', fontsize=13)
ax.set_ylabel('z focal position [µm]', fontsize=13)
ax.set_title('Focal positions: Measured vs Expected', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
out_path = "slm_output_paraxial/tweezer_focal_positions.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✓ Saved plot: {out_path}")

# Calculate error
errors = np.array(focal_positions) - z_expected
rms_error = np.sqrt(np.mean(errors**2))
max_error = np.max(np.abs(errors))

print(f"\n--- Error Analysis ---")
print(f"  RMS error: {rms_error:.2f} µm")
print(f"  Max error: {max_error:.2f} µm")
print(f"  Z-range: {z_expected.max() - z_expected.min():.2f} µm")

if rms_error < 2.0:
    print(f"\n✓✓✓ SUCCESS! Tweezers focus on tilted plane within {rms_error:.2f} µm accuracy!")
else:
    print(f"\n⚠ Errors are larger than expected")

print("="*70)
plt.close()
