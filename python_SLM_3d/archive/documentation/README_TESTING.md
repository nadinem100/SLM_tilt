# Testing the Paraxial Fix

## Quick Start

### 1. Test with f = 200 mm (the problematic case)
```bash
cd python_SLM_3d
python final_run_visualize_paraxial.py
```

**Expected output:**
- Output directory: `slm_output_paraxial/251106-XXXXXX_f200000um_...`
- Console shows: `Focal plane pixel size: 81.48 µm/pixel`
- Plots show: Tweezers on tilted plane (yellow dots follow magenta line)

### 2. Compare with f = 7 mm
Edit line 44 in [final_run_visualize_paraxial.py](final_run_visualize_paraxial.py):
```python
FOCAL_LENGTH_UM = 7000.0  # Change from 200000.0
```

Then run again:
```bash
python final_run_visualize_paraxial.py
```

**Expected output:**
- Output directory: `slm_output_paraxial/251106-XXXXXX_f7000um_...`
- Console shows: `Focal plane pixel size: 2.85 µm/pixel`
- Plots show: Tweezers on tilted plane (more detailed than f=200mm)

## What to Look For

### X-Z Slice Plot (`*_xz.png`)

#### ✅ Success Indicators:
1. **Yellow dots (actual tweezers) follow the magenta dashed line (ideal tilt)**
   - For 5° tilt, slope should be tan(5°) ≈ 0.087
   - z-range: approximately ±25 µm for a 20×20 array with 30 µm spacing

2. **Intensity pattern shows diagonal bands**
   - Bright regions form a tilted stripe pattern
   - Each vertical column should have peak intensity at different z

3. **Gamma correction makes tweezers visible**
   - Even for f=200mm, you should see distinct vertical intensity stripes
   - Title shows `γ=0.40` (gamma correction applied)

#### ❌ Failure Indicators (old formula):
- Yellow dots cluster near z = 0 (no tilt)
- All intensity concentrated in horizontal band at z = 0
- Magenta line doesn't intersect any tweezers

### Three-Plane Plot (`*_3planes.png`)

Shows intensity at three z-positions: [z_min, 0, z_max]

#### What to expect:
- **Left panel (z_min):** Tweezers on left side bright, right side dim
- **Middle panel (z=0):** Middle tweezers brightest
- **Right panel (z_max):** Tweezers on right side bright, left side dim

This confirms different tweezers focus at different z positions.

## Console Output to Check

```
Focal plane pixel size: XX.XX µm/pixel
Full FOV: XXXXX × XXXXX µm²
ROI extent: XXXX × XXXX µm²
```

### Expected values:

| Focal Length | Pixel Size | Full FOV | ROI |
|-------------|-----------|----------|-----|
| f = 7 mm | ~2.85 µm/px | ~7000 µm | ~1000 µm |
| f = 200 mm | ~81 µm/px | ~200,000 µm | ~3000 µm |

### Tweezer z-positions:
```
Using actual tweezer z positions:
  Range: [-24.9, +24.9] µm
  Mean: 0.0 µm
```

For a 5° tilt with 20 tweezers spanning ~570 µm:
- Expected range: ±tan(5°)×285µm ≈ ±25 µm ✓

## Side-by-Side Comparison

### Run both focal lengths and compare:

```bash
# Terminal 1
python final_run_visualize_paraxial.py  # f=200mm

# Edit file to change FOCAL_LENGTH_UM = 7000.0

# Terminal 1 again
python final_run_visualize_paraxial.py  # f=7mm
```

### Open both X-Z plots and compare:

**f = 7 mm:**
- X-axis: ±400 µm (tweezers span ~600 µm)
- Z-axis: ±50 µm
- Individual tweezers: Clearly resolved (spacing 30 µm > pixel size 2.85 µm)
- Tilt line: Visible, matches yellow dots

**f = 200 mm:**
- X-axis: ±300 µm (tweezers span ~600 µm)
- Z-axis: ±50 µm
- Individual tweezers: Blurred (spacing 30 µm < pixel size 81 µm)
- Tilt line: Still visible as diagonal intensity pattern, matches tweezers

**Key Point:** Even though f=200mm has worse resolution, the **tilt should still be visible**!

## Troubleshooting

### Problem: "Plots look almost identical to old runs"

**Check:**
1. Are you running `final_run_visualize_paraxial.py` (not `final_run_visualize.py`)?
2. Does line 25 say `from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers`?
3. Is the output directory `slm_output_paraxial/` (not `slm_output/`)?

### Problem: "Tweezers are too dim to see"

**Fix:** Lower gamma correction in line 70:
```python
GAMMA_CORRECTION = 0.3  # Try 0.25-0.35 for very dim cases
```

### Problem: "Plot is oversaturated"

**Fix:** Increase gamma in line 70:
```python
GAMMA_CORRECTION = 0.5  # Try 0.5-0.6 for bright cases
```

### Problem: "ImportError: No module named slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial"

**Fix:** Make sure you're in the right directory:
```bash
cd "/Users/nadinemeister/Library/CloudStorage/Box-Box/EndresLab/z_Second Experiment/Code/SLM simulation/nadine/DMD_SLM/python_SLM_3d"
ls -la slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial.py  # Should exist!
```

### Problem: "Phase pattern looks wrong"

**Note:** The paraxial formula only changes the **internal GS algorithm**, not the final phase mask much. The big difference is in the **convergence** - tweezers should now focus at the correct z-positions.

To verify:
1. Check X-Z plot: tweezers follow tilt line?
2. Check console output: error signal converges to <0.01?
3. Compare old vs new phase masks: They may look similar, but produce different results!

## Validation Checklist

- [ ] Run `final_run_visualize_paraxial.py` with `FOCAL_LENGTH_UM = 200000.0`
- [ ] Check console output: pixel size ~81 µm/px?
- [ ] Check X-Z plot: yellow dots follow magenta line?
- [ ] Check output directory: `slm_output_paraxial/`?
- [ ] Run again with `FOCAL_LENGTH_UM = 7000.0`
- [ ] Check console output: pixel size ~2.85 µm/px?
- [ ] Check X-Z plot: yellow dots still follow magenta line?
- [ ] Compare both: tilt works for both focal lengths?

If all boxes checked: ✅ **Paraxial fix is working!**

## Next Steps

1. **Experiment with different tilts:**
   ```python
   TILT_ANGLE_X = 10  # Line 48 - try 3, 5, 10, 15 degrees
   ```

2. **Try different focal lengths:**
   ```python
   FOCAL_LENGTH_UM = 100000.0  # 100 mm
   FOCAL_LENGTH_UM = 50000.0   # 50 mm
   ```
   All should work now with paraxial formula!

3. **Adjust visualization for your screen:**
   ```python
   GAMMA_CORRECTION = 0.35   # Line 70 - adjust for visibility
   DPI = 150                 # Line 54 - lower for faster, higher for publication
   ```

4. **Use the phase masks:**
   - BMP files are in the output directory
   - Ready to load onto your SLM
   - Should produce tilted tweezers for any focal length!

## Questions?

See these files for more details:
- [PARAXIAL_FIX_EXPLANATION.md](PARAXIAL_FIX_EXPLANATION.md) - Physics and math
- [VISUALIZATION_IMPROVEMENTS.md](VISUALIZATION_IMPROVEMENTS.md) - Plotting details
- Comments in `final_run_visualize_paraxial.py` - Code documentation
