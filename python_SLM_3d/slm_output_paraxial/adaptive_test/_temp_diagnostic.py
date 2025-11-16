
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift
from pathlib import Path
import yaml
from scipy.ndimage import maximum_filter

PKL_PATH = "slm_output_paraxial/adaptive_test/_adaptive_5x5_tilt20deg_20251108_110609.pkl"
FOCAL_LENGTH_UM = 200000.0
WAVELENGTH_UM = 0.689
Z_RANGE_UM = 100.0
Z_STEPS = 51
N_TWEEZERS_TO_SHOW = 5

print("\nLoading results and generating diagnostic...")

# Load pickle
with open(PKL_PATH, 'rb') as f:
    bundle = pickle.load(f)

phase_mask = bundle.phase_mask
A_in = bundle.A_in

# Load SLM parameters
with open("../slm_parameters.yml", 'r') as f:
    config = yaml.safe_load(f)
pixel_um = config['slm_parameters']['pixel_um']

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

# Pupil coordinates
yy = (np.arange(H) - H/2) * pixel_um_padded
xx = (np.arange(W) - W/2) * pixel_um_padded
X_pupil, Y_pupil = np.meshgrid(xx, yy)
R2_pupil = X_pupil**2 + Y_pupil**2

# Find tweezers at z=0
A_focal_0 = fftshift(fft2(ifftshift(A_pupil)))
I_focal_0 = np.abs(A_focal_0)**2

I_maxfilter = maximum_filter(I_focal_0, size=20)
peaks = (I_focal_0 == I_maxfilter) & (I_focal_0 > 0.1 * I_focal_0.max())
peak_coords = np.where(peaks)

tweezer_positions = []
for iy, ix in zip(peak_coords[0], peak_coords[1]):
    tweezer_positions.append((ix, iy, x_focal_um[ix], y_focal_um[iy]))

if len(tweezer_positions) > N_TWEEZERS_TO_SHOW:
    indices = np.linspace(0, len(tweezer_positions)-1, N_TWEEZERS_TO_SHOW, dtype=int)
    selected_tweezers = [tweezer_positions[i] for i in indices]
else:
    selected_tweezers = tweezer_positions

# Build z-scan
z_scan = np.linspace(-Z_RANGE_UM, +Z_RANGE_UM, Z_STEPS)
I_curves = np.zeros((len(selected_tweezers), Z_STEPS))

for zi, z_test in enumerate(z_scan):
    phase_defocus = +(k / (2.0 * f * f)) * z_test * R2_pupil
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    I_focal = np.abs(A_out)**2

    for tw_idx, (ix, iy, x_um, y_um) in enumerate(selected_tweezers):
        box_size = 3
        iy_min = max(0, iy - box_size//2)
        iy_max = min(H, iy + box_size//2 + 1)
        ix_min = max(0, ix - box_size//2)
        ix_max = min(W, ix + box_size//2 + 1)
        I_curves[tw_idx, zi] = np.max(I_focal[iy_min:iy_max, ix_min:ix_max])

# Create figure
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[1.2, 1, 1, 1])

# Left panel: X-Y focal plane
ax_xy = fig.add_subplot(gs[:, 0])
I_focal_norm = I_focal_0 / I_focal_0.max()

if len(tweezer_positions) > 0:
    all_x = [x for _, _, x, _ in tweezer_positions]
    all_y = [y for _, _, _, y in tweezer_positions]
    x_min, x_max = min(all_x) - 100, max(all_x) + 100
    y_min, y_max = min(all_y) - 100, max(all_y) + 100

    ix_min = np.argmin(np.abs(x_focal_um - x_min))
    ix_max = np.argmin(np.abs(x_focal_um - x_max))
    iy_min = np.argmin(np.abs(y_focal_um - y_min))
    iy_max = np.argmin(np.abs(y_focal_um - y_max))

    I_crop_xy = I_focal_norm[iy_min:iy_max, ix_min:ix_max]
    extent_xy = [x_focal_um[ix_min], x_focal_um[ix_max], y_focal_um[iy_min], y_focal_um[iy_max]]
else:
    I_crop_xy = I_focal_norm
    extent_xy = [x_focal_um.min(), x_focal_um.max(), y_focal_um.min(), y_focal_um.max()]

ax_xy.imshow(I_crop_xy, origin='lower', extent=extent_xy, cmap='gray', vmin=0, vmax=1)

for idx, (ix, iy, x_um, y_um) in enumerate(tweezer_positions):
    ax_xy.plot(x_um, y_um, 'o', color='cyan', markersize=8, markeredgecolor='white', markeredgewidth=1.5, alpha=0.6)

colors = plt.cm.tab10(np.linspace(0, 1, len(selected_tweezers)))
for tw_idx, (ix, iy, x_um, y_um) in enumerate(selected_tweezers):
    ax_xy.plot(x_um, y_um, 'o', color=colors[tw_idx], markersize=14, markeredgecolor='white', markeredgewidth=2)
    ax_xy.text(x_um, y_um, str(tw_idx+1), color='white', fontsize=12, fontweight='bold', ha='center', va='center')

ax_xy.set_xlabel('x [µm]', fontsize=13, fontweight='bold')
ax_xy.set_ylabel('y [µm]', fontsize=13, fontweight='bold')
ax_xy.set_title('Focal Plane (z=0)\\nNumbered = analyzed tweezers', fontsize=14, fontweight='bold')
ax_xy.set_aspect('equal')

# Right panels: Z-profiles
for tw_idx in range(min(len(selected_tweezers), 6)):
    if tw_idx < len(selected_tweezers):
        row = tw_idx // 3
        col = 1 + (tw_idx % 3)
        ax = fig.add_subplot(gs[row, col])

        ix, iy, x_um, y_um = selected_tweezers[tw_idx]
        intensities = I_curves[tw_idx, :]
        intensities_norm = intensities / intensities.max()

        peak_idx = np.argmax(intensities)
        peak_z = z_scan[peak_idx]
        sharpness = intensities[peak_idx] / np.mean(intensities)

        ax.plot(z_scan, intensities_norm, linewidth=2.5, color=colors[tw_idx])
        ax.axvline(peak_z, color='red', linestyle='--', linewidth=2, label=f'Peak @ {peak_z:+.1f} µm')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='cyan', linestyle=':', alpha=0.5, label='z=0')

        ax.set_xlabel('z [µm]', fontsize=11)
        ax.set_ylabel('Norm. Intensity', fontsize=11)
        ax.set_title(f'Tweezer #{tw_idx+1}\\n({x_um:+.0f}, {y_um:+.0f}) µm, Sharp={sharpness:.2f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(0, 1.1)

plt.tight_layout()

out_path = Path(PKL_PATH).parent / "tweezer_xz_profiles.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"✓ Saved diagnostic: {out_path}")
plt.close()
