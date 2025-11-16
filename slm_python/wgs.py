import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
import time
import matplotlib.pyplot as plt

# ----------------------
# Global parameters
# ----------------------
n_horiz = 50
n_vert = 50
x_pixels = 4000
y_pixels = 2464
pixel_um = 3.74         # μm per pixel
waist_um = 9/2 * 1e3     # Physical beam waist in micrometers
spacing_factor = 7       # spacing_factor * w0 is spacing between tweezers

# Holoeye SLM parameters
fill_factor = 0.90       # fractional active area
eflectivity = 0.60        # reflection coefficient
spacing_global = 0.5 * (n_horiz + 2)

# Gerchberg–Saxton parameters
useFilter = False
useGPU = False   # unimplemented
iterations = 10
gain = 1.0
height_corr = np.ones(n_horiz * n_vert)
pos_x_feedback_master = np.zeros(n_horiz * n_vert)
pos_y_feedback_master = np.zeros(n_horiz * n_vert)
first_run = True
position_correction = 0

# Over-sampling factor for grid
scal = 4
x_pixel_size = scal
y_pixel_size = scal

# Define pixel coordinates
# MATLAB: 1:x_pixels; Python: 0..x_pixels-1 for indices
# but for physical grid we use 1-based values
x0 = np.arange(1, x_pixels + 1)
y0 = np.arange(1, y_pixels + 1)
x = np.arange(1, x_pixel_size * x_pixels + 1)
y = np.arange(1, y_pixel_size * y_pixels + 1)

# Create SLM and WGS grids
X0, Y0 = np.meshgrid(x0, y0)
X,  Y  = np.meshgrid(x,  y)
# radial coordinate (unused later)
R = np.round(np.sqrt((X - len(x)/2)**2 + (Y - len(y)/2)**2)).astype(int)

# Input beam amplitude (Gaussian)
waist_in = waist_um / pixel_um
A_in = np.zeros((len(y), len(x)), dtype=np.complex128)
# center indices for injection region
y_mid = y_pixels // 2
x_mid = x_pixels // 2
y_start = (scal - 1) * y_mid
y_end   = (scal + 1) * y_mid
x_start = (scal - 1) * x_mid
x_end   = (scal + 1) * x_mid
# fill central region with Gaussian amplitude
X0c = X0 - x_mid
Y0c = Y0 - y_mid
A_in[y_start:y_end, x_start:x_end] = np.exp(-(X0c**2 + Y0c**2) / waist_in**2)

# circular aperture mask
pad = ((np.abs(X - scal * x_pixels / 2) < x_pixels/2) &
       (np.abs(Y - scal * y_pixels / 2) < y_pixels/2)).astype(float)

# Compute single‑tweezer PSF via FFT
tmp = ifftshift(A_in)
A_single = fftshift(fft2(tmp))

# find 1/e^2 radius via power spectrum
Ps = np.abs(A_single)**2
P_flat = Ps.ravel()
max_idx = int(np.argmax(P_flat))
max_val = P_flat[max_idx]
stop_idx = max_idx
# step until below 1/e^2 of peak
e2 = np.exp(-2) * max_val
while stop_idx < len(P_flat) and P_flat[stop_idx] >= e2:
    stop_idx += 1
# spacing in pixels on atom plane
diam_pix = stop_idx - max_idx
spacing = math.ceil(spacing_factor * 2 * diam_pix)

# Build target array of tweezers
spacing_h = spacing
spacing_v = int(spacing * y_pixels / x_pixels)
h_offset = 0
v_offset = int(-spacing / 2)
# center of PSF
y_ctr, x_ctr = np.unravel_index(max_idx, A_single.shape)

# offsets for each tweezer index
hh = np.arange(1, n_horiz + 1)
vv = np.arange(1, n_vert + 1)
h_offs = np.round(spacing_h * (hh - n_horiz/2)).astype(int)
v_offs = np.round(spacing_v * (vv - n_vert/2)).astype(int)
# grid of target centers
H_grid, V_grid = np.meshgrid(h_offs, v_offs)
target_cols = (x_ctr + h_offset + H_grid).ravel()
target_rows = (y_ctr + v_offset + V_grid).ravel()

# build A_target with inverse height corrections
A_target = np.zeros_like(A_single)
# apply 1/sqrt(height_corr)
hc = height_corr.flatten()
A_target[target_rows, target_cols] = np.sqrt(1.0 / hc)

# precompute neighbor indices for GS feedback
box1 = 1
n_tw = n_horiz * n_vert
coords = []
for idx in range(n_tw):
    vr = np.arange(target_rows[idx] - box1, target_rows[idx] + box1 + 1)
    hc = np.arange(target_cols[idx] - box1, target_cols[idx] + box1 + 1)
    RR, CC = np.meshgrid(vr, hc, indexing='ij')
    lin = np.ravel_multi_index((RR.ravel(), CC.ravel()), A_target.shape)
    coords.append(lin)
coords = np.hstack(coords)
# height correction factors for each sub‑pixel
height_corr2 = np.repeat(1.0 / height_corr.flatten(), (2*box1 + 1)**2)

# Initialize GS phase mask
g = np.ones_like(A_in)
if first_run:
    psi0 = 2 * np.pi * np.random.rand(y_pixels, x_pixels)
    N_cutoff = np.inf
else:
    # placeholder: load previous phase mask
    psi0 = None
    N_cutoff = np.inf
# full-grid phase array
tot_y, tot_x = len(y), len(x)
psi = np.zeros((tot_y, tot_x), dtype=float)
psi[y_start:y_end, x_start:x_end] = psi0

# Freespace propagation parameters for tilted plane
goal_plane_tilt = 5  # degrees
y_len = 9.22e-3
x_len = 15.56e-3
dx = x_len / x_pixels
dy = y_len / y_pixels
xpp = np.linspace(-x_len/2*scal, x_len/2*scal - dx, tot_x)
ypp = np.linspace(-y_len/2*scal, y_len/2*scal - dy, tot_y)
XX, YY = np.meshgrid(xpp, ypp)
f = 200e-3
k = 2 * np.pi / (689e-9)
zss = np.linspace(1, -1, tot_x)
zzz = 3e-3 * np.tile(zss, (tot_y, 1))
Gg = 0.6

# Main GS loop
start = time.time()
for ii in range(1, iterations + 1):
    # forward update
    A_mod = A_in * np.exp(1j * psi)
    A_out = fftshift(fft2(ifftshift(A_mod)))
    if ii <= N_cutoff:
        psi_out = np.mod(np.angle(A_out), 2 * np.pi)
    # tilted-plane propagation
    phase_term = np.exp(-1j * k * zzz * (XX**2 + YY**2) / (2 * f * (f - zzz)))
    B_out = np.abs(fftshift(fft2(A_mod * phase_term)))

    # feedback weight update
    B_flat = B_out.ravel()
    if ii > 1:
        denom = (1 - Gg * (1 - np.sqrt(height_corr2)))
        B_flat[coords] /= denom
        B_out = B_flat.reshape(B_out.shape)

    B_vals = B_flat[coords]
    B_mean = B_vals.mean()
    B_vals_box = B_vals.reshape((2*box1+1)**2, n_tw)
    B_box_mean = B_vals_box.mean(axis=0)
    weight = B_mean / B_box_mean
    # update g
    g_flat = g.ravel()
    g_flat[coords] = np.repeat(weight, (2*box1+1)**2) * g_flat[coords]
    g = g_flat.reshape(g.shape)

    # enforce amplitude constraint & inverse propagation
    print(f"Iteration {ii}: error signal = {np.std(weight):.5f}")
    # compute next psi
    temp = fft2(fftshift(g * A_target)) * np.exp(1j * k * zzz * (XX**2 + YY**2) / (2 * f * (f - zzz)))
    temp = ifftshift(ifft2(temp))
    A_iter = fftshift(temp) * np.exp(1j * psi_out)
    A_new = fftshift(ifft2(ifftshift(A_iter)))
    psi = np.mod(np.angle(A_new), 2 * np.pi) * pad

end = time.time()
print(f"GS complete in {end - start:.2f} s")

# extract final phase mask and save
y_pixels1 = y_pixels
x_pixels1 = x_pixels
phase_mask = psi[y_start:y_end, x_start:x_end]
phase_mask = np.mod(phase_mask, 2 * np.pi)

filename = f"try20z1_n{n_horiz}x{n_vert}_iter{iterations}_spac{spacing_factor}_angle{goal_plane_tilt}"

def save_mask(mask, filename, x_pixels, y_pixels, scal):
    """
    Save the phase mask:
    - NumPy binary (.npy)
    - Text file (.txt)
    - PNG image (.png) with HSV colormap
    """
    np.save(f"{filename}.npy", mask)
    np.savetxt(f"{filename}.txt", mask)
    plt.imsave(f"{filename}.png", mask, cmap='hsv')

save_mask(phase_mask, filename, x_pixels, y_pixels, scal)
