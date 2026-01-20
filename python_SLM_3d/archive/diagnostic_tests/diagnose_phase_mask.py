"""
Diagnostic script: Check if the phase mask actually encodes a tilt.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Load the pickle
PKL_PATH = "slm_output_paraxial/251106-110355_f200000um_sp30.0um_planes5_tilt_5_tw_20/_20x20_tol5.0e-05_v3_20251106_110448.pkl"

with open(PKL_PATH, 'rb') as f:
    bundle = pickle.load(f)

print("="*70)
print("PHASE MASK DIAGNOSTICS")
print("="*70)

phase = bundle.phase_mask
print(f"\nPhase mask shape: {phase.shape}")
print(f"Phase range: [{phase.min():.4f}, {phase.max():.4f}] radians")
print(f"Phase range: [{phase.min()/(2*np.pi):.4f}, {phase.max()/(2*np.pi):.4f}] × 2π")

# For a tilted plane, we expect the phase to vary across the mask
# Check if there's spatial variation
phase_std_per_row = np.std(phase, axis=1)  # Variation along x for each y
phase_std_per_col = np.std(phase, axis=0)  # Variation along y for each x

print(f"\nSpatial variation:")
print(f"  Std across rows (x-direction): mean={np.mean(phase_std_per_row):.4f}, max={np.max(phase_std_per_row):.4f}")
print(f"  Std across cols (y-direction): mean={np.mean(phase_std_per_col):.4f}, max={np.max(phase_std_per_col):.4f}")

# Look at the phase gradient (should show tilt direction)
grad_y, grad_x = np.gradient(phase)
print(f"\nPhase gradients:")
print(f"  Grad in x: mean={np.mean(grad_x):.6f}, std={np.std(grad_x):.6f}")
print(f"  Grad in y: mean={np.mean(grad_y):.6f}, std={np.std(grad_y):.6f}")

# Check for large-scale trends (linear tilt component)
H, W = phase.shape
x = np.arange(W)
y = np.arange(H)

# Average phase vs x (average over all y)
phase_avg_x = np.mean(phase, axis=0)
# Fit linear trend
p_x = np.polyfit(x, phase_avg_x, 1)
print(f"\nLinear trend in x: slope={p_x[0]:.8f} rad/pixel")

# Average phase vs y
phase_avg_y = np.mean(phase, axis=1)
p_y = np.polyfit(y, phase_avg_y, 1)
print(f"Linear trend in y: slope={p_y[0]:.8f} rad/pixel")

# Plot the phase mask
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Raw phase
im1 = axes[0, 0].imshow(phase, cmap='twilight', aspect='auto')
axes[0, 0].set_title('Phase mask (raw)')
axes[0, 0].set_xlabel('x [pixels]')
axes[0, 0].set_ylabel('y [pixels]')
plt.colorbar(im1, ax=axes[0, 0], label='Phase [rad]')

# Phase gradient in x
im2 = axes[0, 1].imshow(grad_x, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
axes[0, 1].set_title('Phase gradient in x')
axes[0, 1].set_xlabel('x [pixels]')
axes[0, 1].set_ylabel('y [pixels]')
plt.colorbar(im2, ax=axes[0, 1], label='∂φ/∂x [rad/px]')

# Phase gradient in y
im3 = axes[1, 0].imshow(grad_y, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
axes[1, 0].set_title('Phase gradient in y')
axes[1, 0].set_xlabel('x [pixels]')
axes[1, 0].set_ylabel('y [pixels]')
plt.colorbar(im3, ax=axes[1, 0], label='∂φ/∂y [rad/px]')

# Average phase profiles
axes[1, 1].plot(x, phase_avg_x, 'b-', label='Avg phase vs x', alpha=0.7)
axes[1, 1].plot(x, np.polyval(p_x, x), 'b--', label=f'Linear fit (slope={p_x[0]:.2e})', linewidth=2)
ax2 = axes[1, 1].twinx()
ax2.plot(y, phase_avg_y, 'r-', label='Avg phase vs y', alpha=0.7)
ax2.plot(y, np.polyval(p_y, y), 'r--', label=f'Linear fit (slope={p_y[0]:.2e})', linewidth=2)
axes[1, 1].set_xlabel('Position [pixels]')
axes[1, 1].set_ylabel('Average phase (x-direction) [rad]', color='b')
ax2.set_ylabel('Average phase (y-direction) [rad]', color='r')
axes[1, 1].legend(loc='upper left')
ax2.legend(loc='upper right')
axes[1, 1].set_title('Average phase profiles')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

out_path = Path(PKL_PATH).parent / "phase_mask_diagnostic.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved diagnostic plot: {out_path}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("="*70)
print("\nFor a 5° tilt along x, we expect:")
print("  - Strong phase gradient in x-direction (to create defocus)")
print("  - Minimal linear trend (defocus is quadratic, not linear)")
print("  - Complex phase pattern (from WGS algorithm)")
print("\nIf the slopes are near zero and the phase looks random:")
print("  → The tilt may not be encoded properly")
print("="*70)
