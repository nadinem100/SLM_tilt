#!/bin/bash
#SBATCH --job-name=slm_80x80_tilt-13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm_logs/slurm_%j.out
#SBATCH --error=../slurm_logs/slurm_%j.err
#SBATCH --mail-user=nmeister@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
mkdir -p $REPO_ROOT/slurm_logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "=========================================="

module purge
module load python/3.11
module load cuda/12.1

source $REPO_ROOT/.venv/bin/activate
cd $SCRIPT_DIR

echo "Python version:"
python --version
echo "GPU info:"
nvidia-smi
echo ""

# Parameters for this run
TILT_ANGLE=-13
N_Z_PLANES=3
ITERATIONS=100
SCAL=4
WAIST_COEFF=9.0

# Create a modified version of test_adaptive_gs.py with these parameters
cat > test_adaptive_gs_temp.py << 'EOFPYTHON'
"""
Test the adaptive multi-plane GS algorithm.
Compares standard vs adaptive versions.
"""

import os
os.environ['PYDEVD_USE_FAST_XML'] = '1'

from pathlib import Path
import numpy as np
from PIL import Image
import time
from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers

# ================================ CONFIG ================================
YAML_PATH = "../slm_parameters.yml"

# Grid configuration (small test for speed)
N_HORIZ = 80
N_VERT = 80
SPACING_UM = 7.2

# GS algorithm
ITERATIONS = ITERATIONS_PLACEHOLDER
GG = 0.6
REDSLM = 1
SCAL = SCAL_PLACEHOLDER
WAIST_UM = WAIST_PLACEHOLDER / 2 * 1e3  # microns
TOL = 5e-3

# Optics
FOCAL_LENGTH_UM = 200000.0  # 200 mm
WAVELENGTH_UM = 0.689

# Tilt configuration
TILT_ANGLE_X = TILT_PLACEHOLDER  # degrees
N_Z_PLANES = N_Z_PLANES_PLACEHOLDER

# Adaptive parameters
Z_SCAN_EVERY = 5  # Scan every 5 iterations
Z_SCAN_RANGE_UM = 50.0  # ±50 µm around target
Z_SCAN_STEPS = 11  # Coarse scan for speed
PEAK_SHARPNESS_THRESHOLD = 2.0
Z_CORRECTION_FACTOR = 0.3  # Gradual correction

# ================================ BMP EXPORT ================================

def add_blazed_grating(phase_mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Add blazed grating with spatial frequencies (fx, fy) to phase mask."""
    H, W = phase_mask.shape
    xx = np.arange(W, dtype=np.float32)
    yy = np.arange(H, dtype=np.float32)
    gr = (2*np.pi*fx*xx)[None, :] + (2*np.pi*fy*yy)[:, None]
    return np.mod(phase_mask + (gr % (2*np.pi)), 2*np.pi).astype(np.float32, copy=False)


def save_phase_bmp(phase: np.ndarray, out_path: Path) -> None:
    """Save phase mask as 8-bit BMP (0-255 maps to 0-2π)."""
    img8 = (np.clip(phase/(2*np.pi), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8, mode="L").save(out_path)

# ================================ MAIN ================================

def main():
    total_start_time = time.time()

    print("="*70)
    print("ADAPTIVE GS TEST")
    print("="*70)

    OUT_DIR = Path("slm_output_paraxial/adaptive_test")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========== SETUP ==========
    print("\n--- Setting up SLM ---")
    setup_start_time = time.time()
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    slm.init_fields(waist_um=WAIST_UM)
    slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM,
                        odd_tw=1, box1=2)
    slm.set_optics(wavelength_um=WAVELENGTH_UM, focal_length_um=FOCAL_LENGTH_UM)

    print(f"\n--- Assigning planes with {TILT_ANGLE_X}° tilt ---")
    slm.assign_planes_from_tilt(tilt_x_deg=TILT_ANGLE_X, n_planes=N_Z_PLANES)

    if hasattr(slm, '_z_planes') and hasattr(slm, '_z_per_spot'):
        print(f"  Z-planes: {slm._z_planes}")
        print(f"  Z per spot range: [{np.min(slm._z_per_spot):.2f}, {np.max(slm._z_per_spot):.2f}] um")
        print(f"  Number of tweezers: {len(slm._z_per_spot)}")

    setup_time = time.time() - setup_start_time
    print(f"\n[TIMING] Setup completed in {setup_time:.2f} seconds")

    # ========== RUN ADAPTIVE GS ==========
    print("\n" + "="*70)
    print("RUNNING ADAPTIVE MULTI-PLANE GS")
    print("="*70)

    gs_start_time = time.time()

    slm.run_gs_multiplane_adaptive(
        iterations=ITERATIONS,
        Gg=GG,
        z_scan_every=Z_SCAN_EVERY,
        z_scan_range_um=Z_SCAN_RANGE_UM,
        z_scan_steps=Z_SCAN_STEPS,
        peak_sharpness_threshold=PEAK_SHARPNESS_THRESHOLD,
        z_correction_factor=Z_CORRECTION_FACTOR,
        verbose=True,
        tol=TOL
    )

    gs_time = time.time() - gs_start_time
    print(f"\n[TIMING] GS algorithm completed in {gs_time:.2f} seconds")

    # ========== SAVE RESULTS ==========
    print("\n--- Saving results ---")
    label = f"_{N_HORIZ}x{N_VERT}_sp{SPACING_UM}um_tilt{TILT_ANGLE_X}deg_{N_Z_PLANES}planes_{ITERATIONS}iter_scal{SCAL}_waist{WAIST_UM/1e3*2:.1f}"
    bundle = slm.save_pickle(out_dir=str(OUT_DIR), label=label)
    print(f"[OK] Saved: {bundle.file}")

    # Print final z-positions
    if hasattr(slm, '_z_per_spot'):
        z_final = slm._z_per_spot
        print(f"\nFinal z-positions after adaptation:")
        print(f"  Range: [{np.min(z_final):.2f}, {np.max(z_final):.2f}] um")
        print(f"  Mean: {np.mean(z_final):.2f} um")
        print(f"  Std: {np.std(z_final):.2f} um")

    # ========== EXPORT BMP WITH BLAZED GRATING ==========
    print("\n--- Exporting BMP with blazed grating ---")

    phase_mask = slm.phase_mask.copy()
    fx, fy = 1.0 / 7.0, 0.0
    phase_blazed = add_blazed_grating(phase_mask, fx=fx, fy=fy)

    pkl_path = Path(bundle.file)
    stem = pkl_path.stem
    out_bmp = pkl_path.parent / f"{stem}_blazepd7.bmp"

    save_phase_bmp(phase_blazed, out_bmp)
    print(f"[OK] Saved BMP: {out_bmp.resolve()}")

    total_time = time.time() - total_start_time

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"Setup time:      {setup_time:8.2f} seconds")
    print(f"GS algorithm:    {gs_time:8.2f} seconds")
    print(f"Total time:      {total_time:8.2f} seconds")
    print(f"\nConfiguration: {N_HORIZ}x{N_VERT} grid, {ITERATIONS} iterations, {N_Z_PLANES} z-planes")
    print("="*70)


if __name__ == "__main__":
    main()
EOFPYTHON

# Replace placeholders with actual values
sed -i "s/ITERATIONS_PLACEHOLDER/$ITERATIONS/g" test_adaptive_gs_temp.py
sed -i "s/SCAL_PLACEHOLDER/$SCAL/g" test_adaptive_gs_temp.py
sed -i "s/WAIST_PLACEHOLDER/$WAIST_COEFF/g" test_adaptive_gs_temp.py
sed -i "s/TILT_PLACEHOLDER/$TILT_ANGLE/g" test_adaptive_gs_temp.py
sed -i "s/N_Z_PLANES_PLACEHOLDER/$N_Z_PLANES/g" test_adaptive_gs_temp.py

echo "Starting SLM simulation..."
echo "Grid: 80x80, Spacing: 7.2um, Tilt: ${TILT_ANGLE}deg"
echo "N_Z_PLANES=$N_Z_PLANES, ITERATIONS=$ITERATIONS, SCAL=$SCAL, WAIST=${WAIST_COEFF}"
python -u test_adaptive_gs_temp.py

# Clean up temp file
rm test_adaptive_gs_temp.py

deactivate
echo "Job completed!"
