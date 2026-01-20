"""
Diagnostic: Show X-Z intensity profiles for individual tweezers.
This shows what the adaptive algorithm "sees" when measuring focal positions.
"""

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift
from pathlib import Path
import yaml
import sys

# Check if running standalone or called from another script
INTERACTIVE = len(sys.argv) == 1 or '--interactive' in sys.argv

if not INTERACTIVE:
    matplotlib.use('Agg')  # Non-interactive backend

# Path to the adaptive GS result
PKL_PATH = "slm_output_paraxial/adaptive_test/_adaptive_5x5_tilt20deg_20251108_110609.pkl"

# Allow override from command line
if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
    PKL_PATH = sys.argv[1]

# Optics parameters
FOCAL_LENGTH_UM = 200000.0
WAVELENGTH_UM = 0.689

# Z-scan parameters (match what adaptive algorithm used)
Z_RANGE_UM = 100.0  # ±100 µm
Z_STEPS = 51  # More steps for smoother curves

# Number of tweezers to show (show a few examples)
N_TWEEZERS_TO_SHOW = 5

print("="*70)
print("TWEEZER X-Z PROFILE DIAGNOSTIC")
print("="*70)

# Load pickle
print(f"\nLoading: {PKL_PATH}")
with open(PKL_PATH, 'rb') as f:
    bundle = pickle.load(f)

phase_mask = bundle.phase_mask
A_in = bundle.A_in

# Load SLM parameters to get tweezer positions
with open("../slm_parameters.yml", 'r') as f:
    config = yaml.safe_load(f)
pixel_um = config['slm_parameters']['pixel_um']

print(f"  Phase mask shape: {phase_mask.shape}")
print(f"  Input field shape: {A_in.shape}")

# Compute focal plane parameters
H, W = A_in.shape
h, w = phase_mask.shape

pixel_um_padded = pixel_um * 4  # SCAL=4
lam = WAVELENGTH_UM
f = FOCAL_LENGTH_UM
k = 2.0 * np.pi / lam
px_focal_um = (lam * f) / (W * pixel_um_padded)

x_focal_um = (np.arange(W) - W/2) * px_focal_um
y_focal_um = (np.arange(H) - H/2) * px_focal_um

# Build full-size phase
psi_full = np.zeros((H, W), dtype=np.float32)
y0 = (H - h) // 2
x0 = (W - w) // 2
psi_full[y0:y0 + h, x0:x0 + w] = phase_mask

# Complex pupil field
A_pupil = (A_in * np.exp(1j * psi_full)).astype(np.complex128)

# Pupil coordinates for defocus
yy = (np.arange(H) - H/2) * pixel_um_padded
xx = (np.arange(W) - W/2) * pixel_um_padded
X_pupil, Y_pupil = np.meshgrid(xx, yy)
R2_pupil = X_pupil**2 + Y_pupil**2

# Find tweezer positions by looking at z=0 focal plane
print("\nFinding tweezer positions...")
A_focal_0 = fftshift(fft2(ifftshift(A_pupil)))
I_focal_0 = np.abs(A_focal_0)**2

# Find peaks
from scipy.ndimage import maximum_filter
I_maxfilter = maximum_filter(I_focal_0, size=20)
peaks = (I_focal_0 == I_maxfilter) & (I_focal_0 > 0.1 * I_focal_0.max())
peak_coords = np.where(peaks)

tweezer_positions = []
for iy, ix in zip(peak_coords[0], peak_coords[1]):
    tweezer_positions.append((ix, iy, x_focal_um[ix], y_focal_um[iy]))

print(f"  Found {len(tweezer_positions)} tweezers")

# Select tweezers from the same row (same y-coordinate)
# Group tweezers by y-coordinate
from collections import defaultdict
y_groups = defaultdict(list)
for tw in tweezer_positions:
    ix, iy, x_um, y_um = tw
    # Round y to nearest 10 µm to group rows
    y_key = round(y_um / 10.0)
    y_groups[y_key].append(tw)

# Find the row with the most tweezers
if len(y_groups) > 0:
    best_row_key = max(y_groups.keys(), key=lambda k: len(y_groups[k]))
    row_tweezers = y_groups[best_row_key]

    # Sort by x-coordinate
    row_tweezers.sort(key=lambda tw: tw[2])  # Sort by x_um

    # Select N_TWEEZERS_TO_SHOW from this row
    if len(row_tweezers) > N_TWEEZERS_TO_SHOW:
        indices = np.linspace(0, len(row_tweezers)-1, N_TWEEZERS_TO_SHOW, dtype=int)
        selected_tweezers = [row_tweezers[i] for i in indices]
    else:
        selected_tweezers = row_tweezers

    print(f"  Analyzing {len(selected_tweezers)} tweezers from same row (y ≈ {best_row_key*10:.0f} µm)")
else:
    selected_tweezers = tweezer_positions[:N_TWEEZERS_TO_SHOW]
    print(f"  Analyzing {len(selected_tweezers)} tweezers")

# Build z-scan
z_scan = np.linspace(-Z_RANGE_UM, +Z_RANGE_UM, Z_STEPS)
print(f"\nScanning z = {z_scan.min():.1f} to {z_scan.max():.1f} µm ({Z_STEPS} steps)...")

# Storage for intensity curves
I_curves = np.zeros((len(selected_tweezers), Z_STEPS))

for zi, z_test in enumerate(z_scan):
    if zi % 10 == 0:
        print(f"  {zi}/{Z_STEPS}...", end='\r')

    # Propagate to this z
    phase_defocus = +(k / (2.0 * f * f)) * z_test * R2_pupil
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    I_focal = np.abs(A_out)**2

    # Extract intensity at each tweezer position
    for tw_idx, (ix, iy, x_um, y_um) in enumerate(selected_tweezers):
        # Small box around tweezer
        box_size = 3
        iy_min = max(0, iy - box_size//2)
        iy_max = min(H, iy + box_size//2 + 1)
        ix_min = max(0, ix - box_size//2)
        ix_max = min(W, ix + box_size//2 + 1)

        I_curves[tw_idx, zi] = np.max(I_focal[iy_min:iy_max, ix_min:ix_max])

print(f"  Done!{' '*30}")

# Analyze curves
print("\nAnalyzing peaks:")
for tw_idx, (ix, iy, x_um, y_um) in enumerate(selected_tweezers):
    intensities = I_curves[tw_idx, :]
    peak_idx = np.argmax(intensities)
    peak_z = z_scan[peak_idx]
    peak_val = intensities[peak_idx]
    mean_val = np.mean(intensities)

    sharpness = peak_val / mean_val

    print(f"  Tweezer {tw_idx+1} @ ({x_um:+6.1f}, {y_um:+6.1f}) µm:")
    print(f"    Peak at z = {peak_z:+6.1f} µm, sharpness = {sharpness:.2f}")

# Create figure with 3 columns: XY map + 2 columns of Z-profiles
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[1.2, 1, 1, 1])

# Left panel: X-Y focal plane showing which tweezers are analyzed
ax_xy = fig.add_subplot(gs[:, 0])
I_focal_norm = I_focal_0 / I_focal_0.max()

# Crop to ROI around tweezers
if len(tweezer_positions) > 0:
    all_x = [x for _, _, x, _ in tweezer_positions]
    all_y = [y for _, _, _, y in tweezer_positions]
    x_min, x_max = min(all_x) - 100, max(all_x) + 100
    y_min, y_max = min(all_y) - 100, max(all_y) + 100

    # Find pixel bounds
    ix_min = np.argmin(np.abs(x_focal_um - x_min))
    ix_max = np.argmin(np.abs(x_focal_um - x_max))
    iy_min = np.argmin(np.abs(y_focal_um - y_min))
    iy_max = np.argmin(np.abs(y_focal_um - y_max))

    I_crop_xy = I_focal_norm[iy_min:iy_max, ix_min:ix_max]
    extent_xy = [x_focal_um[ix_min], x_focal_um[ix_max],
                 y_focal_um[iy_min], y_focal_um[iy_max]]
else:
    I_crop_xy = I_focal_norm
    extent_xy = [x_focal_um.min(), x_focal_um.max(),
                 y_focal_um.min(), y_focal_um.max()]

ax_xy.imshow(I_crop_xy, origin='lower', extent=extent_xy, cmap='gray',
             vmin=0, vmax=1, interpolation='bilinear')

# Mark all tweezers
for idx, (ix, iy, x_um, y_um) in enumerate(tweezer_positions):
    ax_xy.plot(x_um, y_um, 'o', color='cyan', markersize=8, markeredgecolor='white',
               markeredgewidth=1.5, alpha=0.6)

# Highlight selected tweezers with numbers
colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tweezers)))
for tw_idx, (ix, iy, x_um, y_um) in enumerate(selected_tweezers):
    ax_xy.plot(x_um, y_um, 'o', color=colors[tw_idx], markersize=14,
               markeredgecolor='white', markeredgewidth=2)
    ax_xy.text(x_um, y_um, str(tw_idx+1), color='white', fontsize=12,
               fontweight='bold', ha='center', va='center')

ax_xy.set_xlabel('x [µm]', fontsize=13, fontweight='bold')
ax_xy.set_ylabel('y [µm]', fontsize=13, fontweight='bold')
ax_xy.set_title('Focal Plane (z=0)\nNumbered = analyzed tweezers',
                fontsize=14, fontweight='bold')
ax_xy.set_aspect('equal')

# Right panels: Z-profiles for selected tweezers
for tw_idx in range(min(len(selected_tweezers), 6)):
    if tw_idx < len(selected_tweezers):
        row = tw_idx // 3
        col = 1 + (tw_idx % 3)
        ax = fig.add_subplot(gs[row, col])

        ix, iy, x_um, y_um = selected_tweezers[tw_idx]
        intensities = I_curves[tw_idx, :]

        # Normalize
        intensities_norm = intensities / intensities.max()

        # Find peak
        peak_idx = np.argmax(intensities)
        peak_z = z_scan[peak_idx]
        sharpness = intensities[peak_idx] / np.mean(intensities)

        # Plot with matching color
        ax.plot(z_scan, intensities_norm, linewidth=2.5, color=colors[tw_idx])
        ax.axvline(peak_z, color='red', linestyle='--', linewidth=2,
                   label=f'Peak @ {peak_z:+.1f} µm')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='cyan', linestyle=':', alpha=0.5, label='z=0')

        ax.set_xlabel('z [µm]', fontsize=11)
        ax.set_ylabel('Norm. Intensity', fontsize=11)
        ax.set_title(f'Tweezer #{tw_idx+1}\n({x_um:+.0f}, {y_um:+.0f}) µm, Sharp={sharpness:.2f}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(0, 1.1)

plt.tight_layout()

# Use pickle filename as base for output filename
pkl_name = Path(PKL_PATH).stem  # Get filename without extension
out_path = Path(PKL_PATH).parent / f"{pkl_name}_xz_profiles.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✓ Saved: {out_path}")

# ========== SECOND FIGURE: 3 XY Planes at Key Z-Positions ==========
print("\n" + "="*70)
print("GENERATING 3-PLANE XY GRID")
print("="*70)

# Try to get theoretical z-positions from the bundle
if hasattr(bundle, 'z_per_spot') and bundle.z_per_spot is not None:
    z_theory = bundle.z_per_spot
    z_min_theory = np.min(z_theory)
    z_max_theory = np.max(z_theory)
    z_positions = [z_min_theory, 0.0, z_max_theory]
    plane_labels = [f'Min Z (theory)', 'Center (z=0)', f'Max Z (theory)']
    print(f"\nUsing theoretical z-positions from tweezers:")
    print(f"  Z range: [{z_min_theory:.1f}, {z_max_theory:.1f}] µm")
else:
    # Fallback: use z-scan range
    z_positions = [z_scan.min(), 0.0, z_scan.max()]
    plane_labels = ['Min Z', 'Center (z=0)', 'Max Z']
    print(f"\nUsing z-scan range (no theoretical positions in bundle):")

print(f"\nComputing XY planes at 3 z-positions...")
print(f"  Z positions: {z_positions[0]:.1f}, {z_positions[1]:.1f}, {z_positions[2]:.1f} µm")

# Compute intensity at each z-position
I_planes = np.zeros((3, H, W))
for i, z_pos in enumerate(z_positions):
    phase_defocus = +(k / (2.0 * f * f)) * z_pos * R2_pupil
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    I_planes[i, :, :] = np.abs(A_out)**2

print(f"  Done!")

# Create 1x3 grid figure
fig2 = plt.figure(figsize=(21, 7))
gs2 = fig2.add_gridspec(1, 3, hspace=0.2, wspace=0.3)

# Use same ROI as first figure
for plane_idx in range(3):
    ax = fig2.add_subplot(gs2[0, plane_idx])

    # Get intensity for this plane
    I_plane = I_planes[plane_idx, :, :]
    I_plane_norm = I_plane / I_focal_0.max()  # Normalize to z=0 peak

    # Crop to same ROI as first figure
    I_crop_plane = I_plane_norm[iy_min:iy_max, ix_min:ix_max]

    # Plot
    im = ax.imshow(I_crop_plane, origin='lower', extent=extent_xy,
                   cmap='hot', vmin=0, vmax=1, interpolation='bilinear')

    # Title with z-position
    z_val = z_positions[plane_idx]
    ax.set_title(f'{plane_labels[plane_idx]}\nz = {z_val:+.1f} µm',
                fontsize=14, fontweight='bold')

    ax.set_xlabel('x [µm]', fontsize=12, fontweight='bold')
    if plane_idx == 0:
        ax.set_ylabel('y [µm]', fontsize=12, fontweight='bold')

    ax.set_aspect('equal')

# Add overall title
fig2.suptitle('XY Focal Planes at Min/Center/Max Z-Positions\n(Interactive - use toolbar to zoom)',
              fontsize=16, fontweight='bold', y=1.02)

out_path2 = Path(PKL_PATH).parent / f"{pkl_name}_xy_grid.png"
plt.savefig(out_path2, dpi=200, bbox_inches='tight')
print(f"\n✓ Saved: {out_path2}")

# Show both figures interactively only if running standalone
if INTERACTIVE:
    print("\nShowing interactive plots - use toolbar to zoom!")
    print("  Close the plot windows to exit.")
    plt.show()
else:
    plt.close('all')

print("\n" + "="*70)
print("DONE!")
print("="*70)
