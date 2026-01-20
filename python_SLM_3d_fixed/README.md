# Fixed Adaptive Correction Algorithm (Center-Pivot)

This folder contains a **corrected version** of the adaptive multi-plane GS algorithm that properly targets the geometric z-positions. This is the **center-pivot** version where the tilted plane rotates around the center of the tweezer array.

## The Problem (in `python_SLM_3d/`)

The original adaptive algorithm had a subtle but critical flaw:

```python
# Original (WRONG):
z_error = peak_z - z_target
z_target += correction_factor * z_error  # Drift away from geometric goal!
```

This causes the target z-positions to **drift** away from the original geometric values.

**Example:**
- Geometric goal: z = +50 µm
- Measurement: tweezer focuses at z = +73 µm
- Update: z_target = 50 + 0.3×23 = 56.9 µm (moving AWAY from goal!)
- Eventually drifts to some other value, not +50 µm

## The Fix

The corrected algorithm preserves the original geometric targets:

```python
# Save geometric targets (NEVER UPDATED)
self._z_geometric = geometric_calculation()

# Working targets for GS (GETS UPDATED)
self._z_per_spot = z_geometric.copy()

# During adaptive correction:
z_error = peak_z - z_geometric  # Error relative to ORIGINAL goal
z_per_spot = z_geometric - correction_factor * z_error  # Correct TOWARD goal
```

**Fixed example:**
- Geometric goal: z = +50 µm (saved, never changes)
- Measurement: tweezer focuses at z = +73 µm
- Error: +23 µm (too far)
- Update: z_target = 50 - 0.3×23 = 43.1 µm (aim LOWER to compensate)
- Next: GS targets 43.1 µm → actually focuses closer to 50 µm
- Converges: target ≈ 27 µm → **actual focus = 50 µm** ✓

## Configurable Parameters

The following parameters can be configured from the bash script ([run_slm.sh](run_slm.sh)):

- **N_Z_PLANES**: Number of z-planes for tweezers (default: 5)
- **ITERATIONS**: Number of GS iterations (default: 100)
- **SCAL**: Resolution scaling factor - 2 = fast, 4 = accurate (default: 4)
- **WAIST_COEFF**: Beam waist coefficient (default: 2.6, was 9 originally)
  - Actual waist: `WAIST_UM = WAIST_COEFF / 2 * 1e3` microns

## Files

- [test_adaptive_gs.py](test_adaptive_gs.py) - Main test script (uses FIXED algorithm)
- [slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial.py](slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial.py) - FIXED SLM class
- [run_slm.sh](run_slm.sh) - SLURM submission script for cluster
- [diagnose_tweezer_xz_profiles.py](diagnose_tweezer_xz_profiles.py) - Diagnostic visualization script

## Output

Results are saved to: `slm_output_paraxial/adaptive_test_fixed/`

Files are named with all parameters included:
```
_adaptive_{N_HORIZ}x{N_VERT}_tilt{TILT_ANGLE_X}deg_{N_Z_PLANES}planes_{ITERATIONS}iter_scal{SCAL}_waist{WAIST_COEFF}
```

## Usage

**Local:**
```bash
cd python_SLM_3d_fixed
python test_adaptive_gs.py [N_Z_PLANES] [ITERATIONS] [SCAL] [WAIST_COEFF]
```

**Example:**
```bash
python test_adaptive_gs.py 5 100 4 2.6
```

**Cluster:**
```bash
cd python_SLM_3d_fixed
sbatch run_slm.sh
```

## Which Version Should You Use?

- **`python_SLM_3d_fixed/`**: Precise geometric control (use this!)
- **`python_SLM_3d/`**: "Roughly tilted" tweezers (less accurate)
- **`tilt_edge_fixed/`**: Edge-pivot with precise control (for atoms on tilted surface)
- **`tilt_edge/`**: Edge-pivot, less accurate

For precise control of tweezer positions, **use python_SLM_3d_fixed/** (center-pivot) or **tilt_edge_fixed/** (edge-pivot).
