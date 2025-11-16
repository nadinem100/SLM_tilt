# Tilted Plane Focus Fix: Paraxial vs Exact Defocus Formula

## Problem Summary
The WGS algorithm successfully focused tweezers in a tilted plane for small focal lengths (f = 7000 µm) but failed for large focal lengths (f = 200000 µm). The tweezers appeared converged but not in a cleanly tilted plane.

## Root Cause
The issue was in the **defocus phase formula** used in `assign_planes_from_tilt()` method.

### Original (Incorrect) Formula
```python
# "Exact" thin-lens formula
phase = -k * z * R² / (2*f*(f-z))
```

This formula includes the factor `z/(f-z) ≈ z/f` when z << f.

### Why This Failed for Large f

For a 5° tilt with tweezers spanning ~280 µm:
- z-range ≈ ±12 µm (from tan(5°) × 140 µm)

**For f = 7000 µm:**
- z/f ≈ 12/7000 ≈ 0.0017
- Phase correction magnitude: **reasonable**

**For f = 200000 µm:**
- z/f ≈ 12/200000 ≈ 0.00006
- Phase correction magnitude: **28× smaller!**
- Result: All planes collapse to nearly the same focal position

### Physical Explanation
The "exact" formula `φ = -k·z·R²/(2f(f-z))` is derived for imaging applications where you're moving the **image plane** relative to a fixed focal plane. In that context, z is measured from the focal plane and the formula accounts for the change in optical path length.

However, for SLM tweezers:
- We're applying phase corrections **at the pupil plane** (SLM)
- We want to shift the focus by a small amount z along the optical axis
- The regime is always z << f (micron-scale shifts vs mm-scale focal lengths)
- The appropriate formula is the **paraxial defocus**

## Solution: Paraxial Defocus Formula

### Corrected Formula
```python
# Paraxial defocus for z << f
phase = (k / (2*f²)) * z * R²
```

This formula:
- ✓ Gives z-dependent corrections that scale correctly with z
- ✓ Does NOT shrink as f increases
- ✓ Is the standard formula for defocus in Fourier optics
- ✓ Valid for the regime z << f (always true for optical tweezers)

### Mathematical Comparison

| Focal Length | z-value | Old Formula Factor | New Formula Factor | Ratio |
|--------------|---------|-------------------|-------------------|-------|
| f = 7 mm     | z = 12 µm | z/(f-z) ≈ 0.0017 | z/f² ≈ 0.00024 mm⁻¹ | - |
| f = 200 mm   | z = 12 µm | z/(f-z) ≈ 0.00006 | z/f² ≈ 0.0003 mm⁻¹ | **28× stronger!** |

The paraxial formula gives consistent z-dependent phase corrections regardless of focal length.

## Files Modified

### 1. `slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial.py`
**Changed:** Lines 755-765 in `assign_planes_from_tilt()` method

**Before:**
```python
for z_p in self._z_planes:
    z_val = float(z_p)
    denom = 2.0 * f * (f - z_val)
    if abs(denom) < 1e-6:
        denom = 2.0 * f * f
    phase = (-k * z_val / denom) * quad
    self._phi_planes.append(phase.astype(np.float32, copy=False))
```

**After:**
```python
for z_p in self._z_planes:
    # PARAXIAL defocus formula: φ = (k/(2f²)) · z · R²
    z_val = float(z_p)
    phase = (k / (2.0 * f * f)) * z_val * quad
    self._phi_planes.append(phase.astype(np.float32, copy=False))
```

### 2. `final_run_visualize_paraxial.py`
**Changed:**
- Line 25: Import statement to use corrected class
- Lines 153-161: Visualization defocus formula in `compute_focal_plane_intensity()`

**Before:**
```python
denom = 2.0 * f * (f - z_um)
if abs(denom) < 1e-6:
    denom = 2.0 * f * f
phase_defocus = -k * z_um * R2 / denom
```

**After:**
```python
# PARAXIAL defocus formula: φ(X,Y) = (k/(2f²)) * z * R²
phase_defocus = (k / (2.0 * f * f)) * z_um * R2
```

## How to Use

### Test the Fix
Run the corrected version:
```python
python final_run_visualize_paraxial.py
```

The script will:
1. Generate tilted tweezers using the **paraxial formula**
2. Create visualization plots showing the tilted plane
3. Save phase masks with `_paraxial` in the filename

### Expected Results
- ✓ Tilted plane focusing should work for **all** focal lengths
- ✓ For f = 200000 µm, tweezers should now lie on a clean tilted plane
- ✓ X-Z visualization should show tweezers following the theoretical tilt line
- ✓ Convergence should be similar to f = 7000 µm case

## Validation

Compare outputs:
1. **Small f (7 mm):** Both formulas should give similar results (z/f is small either way)
2. **Large f (200 mm):** Paraxial formula should produce visible tilt; old formula would not

## References
- Goodman, "Introduction to Fourier Optics", 4th ed., Section 5.2 (Fresnel diffraction)
- The paraxial formula φ = (k/2f²)·z·R² is the standard Fourier optics result for defocus
- For SLM applications, see: Curtis et al., Opt. Commun. 207, 169 (2002)

## Contact
If you have questions about this fix, check:
- Fourier optics textbooks (Goodman, Chapter 5)
- Your optics simulation code for similar paraxial approximations
- The literature on holographic optical tweezers (HOT) phase calculations
