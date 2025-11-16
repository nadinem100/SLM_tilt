"""
Test: Does the final phase mask contain the defocus corrections?

Theory: If we want a tweezer to focus at z≠0, the SLM phase mask should contain
a quadratic phase term. Let's check if it does.
"""

import numpy as np
from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers
import matplotlib.pyplot as plt

print("="*70)
print("TEST: Does final phase mask contain defocus corrections?")
print("="*70)

# Simple test: 1 tweezer at z=+20 µm
N_HORIZ = 1
N_VERT = 1
FOCAL_LENGTH_UM = 7000.0
TARGET_Z = 20.0  # Want to focus at +20 µm

print(f"\nSetup: Single tweezer, f={FOCAL_LENGTH_UM/1000:.0f}mm, target z={TARGET_Z}µm")

# Create grid with single tweezer at x=0, and assign it to z=+20µm using tilt
# If we use tilt_angle such that tan(angle) * 0 = 20, that won't work (x=0)
# Instead, use a fake tilt with the tweezer offset from center

slm = SLMTweezers(yaml_path="../slm_parameters.yml", redSLM=1, scal=4)
slm.init_fields(waist_um=4500)
slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=50.0, odd_tw=1, box1=2)
slm.set_optics(wavelength_um=0.689, focal_length_um=FOCAL_LENGTH_UM)

# Manually set the z position for this single tweezer
slm._z_per_spot = np.array([TARGET_Z], dtype=np.float32)
slm._z_planes = np.array([TARGET_Z], dtype=np.float32)
slm._members = [[0]]  # Tweezer 0 belongs to plane 0

# Compute the expected phase correction
H, W = slm.A_target.shape
yy = (np.arange(H) - H / 2) * slm.params.pixel_um
xx = (np.arange(W) - W / 2) * slm.params.pixel_um
X, Y = np.meshgrid(xx, yy)
R2 = X**2 + Y**2

k = 2 * np.pi / 0.689
f = FOCAL_LENGTH_UM
phi_expected = (k / (2.0 * f * f)) * TARGET_Z * R2
phi_expected_max = np.max(np.abs(phi_expected))

print(f"\nExpected phase correction:")
print(f"  Formula: φ = (k/2f²) * z * R²")
print(f"  Max |φ| = {phi_expected_max:.2f} rad = {phi_expected_max/(2*np.pi):.3f} × 2π")
print(f"  At pupil edge R={np.sqrt(R2.max())/1000:.2f} mm")

# Build phase planes for GS
slm._pupil_X2 = (X * X).astype(np.float32)
slm._pupil_Y2 = (Y * Y).astype(np.float32)
slm._pupil_XY = (X * Y).astype(np.float32)
slm._k = k
slm._k_over_2f2 = k / (2.0 * f * f)

quad = (slm._pupil_X2 + slm._pupil_Y2).astype(np.float32)
phase = (k / (2.0 * f * f)) * TARGET_Z * quad
slm._phi_planes = [phase.astype(np.float32)]

print(f"\n--- Running GS (20 iterations) ---")
slm.run_gs_multiplane_v3(iterations=20, Gg=0.6, verbose=False, tol=1e-5)

# Analyze final phase mask
final_phase = slm.phase_mask
print(f"\n--- Final phase mask analysis ---")
print(f"  Shape: {final_phase.shape}")
print(f"  Mean: {final_phase.mean():.4f} rad")
print(f"  Std: {final_phase.std():.4f} rad")
print(f"  Range: [{final_phase.min():.4f}, {final_phase.max():.4f}] rad")

# Check if there's a quadratic component
# Fit a quadratic surface to the phase
from scipy.optimize import curve_fit

y_px, x_px = np.meshgrid(np.arange(final_phase.shape[1]), np.arange(final_phase.shape[0]))
x_px_flat = x_px.ravel()
y_px_flat = y_px.ravel()
phase_flat = final_phase.ravel()

# Quadratic model: φ(x,y) = a + b*x + c*y + d*x² + e*y² + f*x*y
def quadratic_model(xy, a, b, c, d, e, f):
    x, y = xy
    return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y

try:
    # Unwrap phase first for better fitting
    phase_unwrapped = np.unwrap(np.unwrap(final_phase, axis=0), axis=1)
    phase_flat_unwrapped = phase_unwrapped.ravel()

    popt, _ = curve_fit(quadratic_model, (x_px_flat, y_px_flat), phase_flat_unwrapped,
                       p0=[0, 0, 0, 0, 0, 0], maxfev=10000)

    a, b, c, d, e, f_coef = popt

    print(f"\n  Quadratic fit: φ = a + b*x + c*y + d*x² + e*y² + f*x*y")
    print(f"    a (offset) = {a:.4f} rad")
    print(f"    b (linear x) = {b:.6f} rad/px")
    print(f"    c (linear y) = {c:.6f} rad/px")
    print(f"    d (quadratic x²) = {d:.8f} rad/px²")
    print(f"    e (quadratic y²) = {e:.8f} rad/px²")
    print(f"    f (cross x*y) = {f_coef:.8f} rad/px²")

    # Expected quadratic coefficient in pixel coordinates
    # φ = (k/2f²) * z * (x² + y²) where x,y in µm
    # φ = (k/2f²) * z * (px_um * x_px)²
    # φ = [(k/2f²) * z * px_um²] * x_px²
    px_um = slm.params.pixel_um
    expected_quad_coeff = (k / (2.0 * f * f)) * TARGET_Z * (px_um ** 2)

    print(f"\n  Expected quadratic coefficient:")
    print(f"    (k/2f²)*z*px² = {expected_quad_coeff:.8f} rad/px²")

    print(f"\n  Comparison:")
    print(f"    Measured d: {d:.8f}")
    print(f"    Expected:   {expected_quad_coeff:.8f}")
    print(f"    Ratio d/expected: {d/expected_quad_coeff:.3f}")

    if abs(d/expected_quad_coeff) > 0.5:
        print(f"\n  ✓ Final phase CONTAINS significant quadratic term!")
        print(f"    The defocus correction IS in the final mask.")
    else:
        print(f"\n  ✗ Final phase has weak/missing quadratic term.")
        print(f"    The defocus correction may be getting lost in GS.")

except Exception as e:
    print(f"\n  Warning: Could not fit quadratic model: {e}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Raw final phase
im0 = axes[0].imshow(final_phase, cmap='twilight', origin='lower')
axes[0].set_title('Final phase mask')
axes[0].set_xlabel('x [px]')
axes[0].set_ylabel('y [px]')
plt.colorbar(im0, ax=axes[0], label='Phase [rad]')

# Expected phase (just the quadratic part, on SLM region)
h, w = final_phase.shape
phase_expected_slm = phi_expected[H//2-h//2:H//2+h//2, W//2-w//2:W//2+w//2]
im1 = axes[1].imshow(phase_expected_slm, cmap='twilight', origin='lower')
axes[1].set_title(f'Expected defocus (z={TARGET_Z}µm)')
axes[1].set_xlabel('x [px]')
axes[1].set_ylabel('y [px]')
plt.colorbar(im1, ax=axes[1], label='Phase [rad]')

# Difference
diff = final_phase - (phase_expected_slm % (2*np.pi))
im2 = axes[2].imshow(diff, cmap='RdBu_r', origin='lower', vmin=-np.pi, vmax=np.pi)
axes[2].set_title('Difference (final - expected)')
axes[2].set_xlabel('x [px]')
axes[2].set_ylabel('y [px]')
plt.colorbar(im2, ax=axes[2], label='Phase [rad]')

plt.tight_layout()
plt.savefig('slm_output_paraxial/phase_mask_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot: slm_output_paraxial/phase_mask_analysis.png")

print("="*70)
