"""
Interactive Defocus Viewer
Load a phase mask and interactively sweep through z-planes to see where tweezers focus.
Uses slider to add defocus (quadratic phase) without re-running GS.

Works at reduced resolution for interactive speed.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
from pathlib import Path
import sys

# ================================ PARAMETERS ================================

# Default BMP path (can be overridden by command line)
# Latest 80x80 file:
DEFAULT_BMP = r"c:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\slm_output_paraxial\adaptive_test_fixed\_adaptive_80x80_tilt-13deg_3planes_10iter_scal4_waist9.0_20260120_011439_blazepd7.bmp"

# OLD files for reference:
# DEFAULT_BMP = r"C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\python_SLM_3d_fixed\slm_output_paraxial\adaptive_test_matlab_spacing\_matlab_spacing_100x100_tilt-13deg_5planes_50iter_scal4_waist8.1_20260119_231914_blazepd7.bmp"
# DEFAULT_BMP = r"C:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\python_SLM_3d_fixed\slm_output_paraxial\adaptive_test_fixed\_adaptive_20x20_tilt-13deg_10planes_100iter_scal4_waist9.0_20251126_151206_blazepd7.bmp"

# Optical parameters
FOCAL_LENGTH_UM = 200000.0  # 200 mm
WAVELENGTH_UM = 0.689
PIXEL_UM = 3.74  # SLM pixel size
WAIST_COEFF = 8.1  # Beam waist coefficient (mm)

# Z-scan range
Z_MIN = -10000  # um
Z_MAX = 10000   # um
Z_INIT = 0     # Initial z position

# ================================ FUNCTIONS ================================

def load_phase_bmp(bmp_path):
    """Load phase mask from BMP file (0-255 -> 0-2pi)."""
    img = Image.open(bmp_path)
    phase = np.array(img, dtype=np.float32) / 255.0 * 2 * np.pi
    return phase

def compute_focal_plane(A_in, phase_mask, z_um, k, f, X2_Y2):
    """Compute focal plane intensity at defocus z using numpy FFT."""
    # Add defocus phase to the mask
    phase_defocus = (k / (2.0 * f * f)) * z_um * X2_Y2
    total_phase = phase_mask + phase_defocus
    
    # Modulate input beam
    A_mod = A_in * np.exp(1j * total_phase)
    
    # FFT to focal plane (use numpy directly for speed)
    A_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A_mod)))
    I_focal = np.abs(A_focal)**2
    
    return I_focal

# ================================ MAIN ================================

def main():
    # Get BMP path from command line or use default
    if len(sys.argv) > 1:
        bmp_path = sys.argv[1]
    else:
        bmp_path = DEFAULT_BMP
    
    print("="*70)
    print("INTERACTIVE DEFOCUS VIEWER")
    print("="*70)
    print(f"\nLoading: {bmp_path}")
    
    # Check file exists
    if not Path(bmp_path).exists():
        print(f"ERROR: File not found: {bmp_path}")
        return
    
    # Load phase mask
    phase_mask_full = load_phase_bmp(bmp_path)
    h_full, w_full = phase_mask_full.shape
    print(f"  Full phase mask: {h_full} x {w_full}")
    
    # Use full resolution (no downsampling to avoid phase corruption)
    downsample = 1  # Changed from 2 - downsampling corrupts wrapped phase!
    phase_mask = phase_mask_full
    h, w = phase_mask.shape
    print(f"  Using full resolution: {h} x {w}")
    
    # Effective pixel size after downsampling
    pixel_um = PIXEL_UM * downsample
    
    # Create input beam (Gaussian)
    waist_um = WAIST_COEFF / 2 * 1000  # Convert to um
    waist_pixels = waist_um / pixel_um
    
    y = np.arange(h) - h/2
    x = np.arange(w) - w/2
    X, Y = np.meshgrid(x, y)
    
    A_in = np.exp(-(X**2 + Y**2) / waist_pixels**2).astype(np.float32)
    print(f"  Input beam waist: {waist_um:.0f} um ({waist_pixels:.1f} pixels)")
    
    # Pupil coordinates for defocus (in um)
    X_um = X * pixel_um
    Y_um = Y * pixel_um
    X2_Y2 = X_um**2 + Y_um**2
    
    # Optical parameters
    k = 2 * np.pi / WAVELENGTH_UM
    f = FOCAL_LENGTH_UM
    
    # Focal plane coordinates
    px_focal = (WAVELENGTH_UM * f) / (w * pixel_um)
    py_focal = (WAVELENGTH_UM * f) / (h * pixel_um)
    x_focal = (np.arange(w) - w/2) * px_focal
    y_focal = (np.arange(h) - h/2) * py_focal
    
    print(f"  Focal plane pixel: {px_focal:.2f} x {py_focal:.2f} um")
    
    # Compute initial focal plane
    print("\nComputing initial focal plane (z=0)...")
    I_init = compute_focal_plane(A_in, phase_mask, 0, k, f, X2_Y2)
    print("  Done!")
    
    # Find region with tweezers - use higher threshold for tighter crop
    threshold = 0.05 * I_init.max()  # 5% of max
    rows_with_signal = np.any(I_init > threshold, axis=1)
    cols_with_signal = np.any(I_init > threshold, axis=0)
    
    if np.any(rows_with_signal) and np.any(cols_with_signal):
        row_min = np.argmax(rows_with_signal)
        row_max = h - np.argmax(rows_with_signal[::-1])
        col_min = np.argmax(cols_with_signal)
        col_max = w - np.argmax(cols_with_signal[::-1])
        
        # Small margin (5% of region size)
        margin_y = max(5, int(0.05 * (row_max - row_min)))
        margin_x = max(5, int(0.05 * (col_max - col_min)))
        row_min = max(0, row_min - margin_y)
        row_max = min(h, row_max + margin_y)
        col_min = max(0, col_min - margin_x)
        col_max = min(w, col_max + margin_x)
    else:
        row_min, row_max = h//4, 3*h//4
        col_min, col_max = w//4, 3*w//4
    
    print(f"  ROI: [{row_min}:{row_max}, {col_min}:{col_max}]")
    print(f"  ROI size: {row_max-row_min} x {col_max-col_min} pixels")
    
    # Physical extent
    x_extent = [x_focal[col_min], x_focal[col_max-1]]
    y_extent = [y_focal[row_max-1], y_focal[row_min]]
    
    # ================================ CREATE GUI ================================
    
    print("\nCreating interactive viewer...")
    print("  Use slider to change z-plane (defocus)")
    print("  Arrow keys: left/right = +/- 50 um")
    print("  Press 'r' to reset to z=0")
    print("  Close window to exit\n")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    
    # Initial plot
    I_crop = I_init[row_min:row_max, col_min:col_max]
    vmax = np.percentile(I_crop, 99.5)
    
    im = ax.imshow(I_crop, cmap='hot', vmin=0, vmax=vmax,
                   extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                   aspect='equal', interpolation='nearest')
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    title = ax.set_title('Focal plane at z = 0 um')
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='Intensity')
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'z (um)', Z_MIN, Z_MAX, valinit=Z_INIT, valstep=10)
    
    # Create reset button
    ax_reset = plt.axes([0.85, 0.05, 0.1, 0.03])
    button_reset = Button(ax_reset, 'Reset')
    
    # Update function
    def update(val):
        z_um = slider.val
        I_new = compute_focal_plane(A_in, phase_mask, z_um, k, f, X2_Y2)
        I_crop = I_new[row_min:row_max, col_min:col_max]
        im.set_data(I_crop)
        title.set_text(f'Focal plane at z = {z_um:.0f} um')
        fig.canvas.draw_idle()
    
    def reset(event):
        slider.set_val(0)
    
    slider.on_changed(update)
    button_reset.on_clicked(reset)
    
    # Key press handler
    def on_key(event):
        if event.key == 'r':
            slider.set_val(0)
        elif event.key == 'left':
            slider.set_val(max(Z_MIN, slider.val - 50))
        elif event.key == 'right':
            slider.set_val(min(Z_MAX, slider.val + 50))
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()
    print("Viewer closed.")

if __name__ == "__main__":
    main()
