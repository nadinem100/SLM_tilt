# Visualization Improvements for Large Focal Length

## Problems Fixed

### 1. **Output Directory**
- **Old:** Everything saved to crowded `slm_output/`
- **New:** Paraxial runs saved to `slm_output_paraxial/`

### 2. **Intensity Visibility**
- **Problem:** For large f, tweezers appear very dim because intensity spreads over larger area
- **Solution:** Added **gamma correction** (γ = 0.4) to brighten dim features
- `I_display = I^γ` where γ < 1 compresses the intensity range and makes dim spots visible

### 3. **Diagnostic Output**
Added printouts to understand scale issues:
```
Focal plane pixel size: 81.48 µm/pixel  (for f=200mm)
vs
Focal plane pixel size: 2.85 µm/pixel   (for f=7mm)
```

This explains why f=200mm plots look "zoomed out" - each pixel covers 28× more area!

## How to Use

### Run with improved visualization:
```bash
cd python_SLM_3d
python final_run_visualize_paraxial.py
```

### Key Configuration Options

In [final_run_visualize_paraxial.py](final_run_visualize_paraxial.py):

```python
# Line 44: Change focal length
FOCAL_LENGTH_UM = 200000.0  # or 7000.0 for comparison

# Line 70-71: Adjust visualization
GAMMA_CORRECTION = 0.4      # Lower = brighter (try 0.3-0.6)
VMAX_PERCENTILE = 99.5      # Clip brightest pixels for contrast
```

## What Changed in the Code

### 1. Output directory (line 79)
```python
return Path(
    "slm_output_paraxial/"  # New directory!
    f"{ts}_f{FOCAL_LENGTH_UM:.0f}um..."
)
```

### 2. Gamma correction in plotting (line 377)
```python
# Apply gamma correction to improve visibility
xz_display = np.power(np.clip(xz_intensity, 0, 1), gamma)
```

This transformation:
- Makes dim tweezers visible without saturating bright ones
- Similar to monitor/camera gamma correction
- γ = 0.4 means `I_display = I^0.4 ≈ sqrt(sqrt(I))`

### 3. Diagnostic output (lines 493-494, 513-514)
```python
print(f"  Focal plane pixel size: {px_focal_um:.2f} µm/pixel")
print(f"  ROI extent: {x_roi_extent:.0f} × {y_roi_extent:.0f} µm²")
```

## Expected Results

### For f = 7 mm:
- Pixel size: ~2.85 µm/pixel
- ROI: ~1000 × 1000 µm²
- Individual tweezers clearly visible
- Tilt line matches actual tweezers

### For f = 200 mm:
- Pixel size: ~81 µm/pixel
- ROI: ~30,000 × 30,000 µm² (28× larger!)
- **After gamma correction:** Individual tweezers now visible
- **After paraxial fix:** Tilt line matches actual tweezers

## Understanding the Plots

### X-Z Slice Plot
- **X-axis:** Position across the tweezer array
- **Z-axis:** Defocus distance from nominal focal plane
- **Colors:**
  - Cyan dashed line: z = 0 (nominal focus)
  - Magenta dashed line: Ideal tilt plane
  - Yellow dots: Actual tweezer positions (middle row only, y ≈ 0)
  - Orange line: Connects actual tweezers

### What Good Results Look Like
✓ Yellow dots align with magenta line (tweezers on tilted plane)
✓ Intensity maxima form diagonal pattern following the tilt
✓ Each tweezer is visible as a vertical stripe at its (x, z) position

### What Bad Results Look Like (old formula)
✗ Yellow dots cluster near z = 0 (no tilt)
✗ All intensity concentrated at z = 0
✗ Magenta line doesn't pass through any tweezers

## Comparison: Old vs New

| Property | Old (Exact Formula) | New (Paraxial Formula) |
|----------|-------------------|----------------------|
| Formula | φ = -k·z·R²/(2f(f-z)) | φ = (k/2f²)·z·R² |
| f = 7 mm | Works ✓ | Works ✓ |
| f = 200 mm | **Fails** (tweezers at z≈0) | **Works** ✓ |
| Scaling | Corrections ∝ z/f | Corrections ∝ z (correct!) |
| Visualization | Dim, hard to see | Gamma-corrected, clear ✓ |
| Output | Crowded `slm_output/` | Clean `slm_output_paraxial/` |

## Troubleshooting

### "Tweezers still look dim"
→ Decrease `GAMMA_CORRECTION` (try 0.3 or 0.25)

### "Plot is too saturated/bright"
→ Increase `GAMMA_CORRECTION` (try 0.5 or 0.6)

### "X-axis range is wrong"
→ Check `ROI_PAD_UM` (line 60) - increase for more context

### "Can't see individual tweezers"
→ This is expected if focal-plane pixel size >> tweezer spacing
→ For f=200mm: 81 µm/px vs 30 µm spacing means tweezers blend together
→ The tilt pattern should still be visible as diagonal intensity bands

## Technical Notes

### Why Gamma Correction?
Human perception of brightness is nonlinear (approximately logarithmic). Gamma correction compensates:
- **γ < 1:** Compresses bright values, expands dim values → better for seeing faint features
- **γ = 1:** Linear (no correction)
- **γ > 1:** Expands bright values, compresses dim values → emphasizes brightest features

### Focal Plane Pixel Size Formula
```
px_focal = λ·f / (W·px_SLM)
```
Where:
- λ = wavelength (0.689 µm)
- f = focal length (7000 or 200000 µm)
- W = SLM pupil width in pixels (after SCAL padding)
- px_SLM = SLM pixel size (8 µm for typical devices)

This is the diffraction-limited resolution at the focal plane.

### Why Does Resolution Scale With f?
- Larger f → smaller convergence angle θ ≈ r/f
- Smaller θ → larger Airy disk (spot size)
- Spot size ∝ λ·f/D where D is pupil diameter
- In Fourier optics: pixel_size_focal = λ·f/(N·pixel_size_pupil)

For f = 200 mm, tweezers are still well-focused, they just appear at lower resolution in the image plane.
