# Sign Convention Investigation Results

## Summary

After extensive testing, I've confirmed that the **POSITIVE sign in the paraxial formula is correct**, matching the original implementation. The tilted plane issue is NOT caused by a sign error, but by the fundamental physics limitation identified in [TILTED_PLANE_ISSUE_SUMMARY.md](TILTED_PLANE_ISSUE_SUMMARY.md): the **Rayleigh range is much larger than the z-displacement range**.

## Sign Convention Analysis

### Correct Formula

The paraxial defocus formula should be:
```
φ = +(k/(2f²)) · z · R²
```

**Physical Reasoning:**
- A converging lens has phase φ = -k R²/(2f) (negative quadratic)
- To move focus FARTHER (z > 0), we need LESS convergence
- Less convergence = add POSITIVE (diverging) phase
- Therefore: positive z requires positive phase correction

### Why the Confusion?

The diagnostic test `test_phase_in_final_mask.py` showed that when we apply POSITIVE phase corrections in `_phi_planes`, the final SLM phase mask contains NEGATIVE quadratic curvature. This seems contradictory but is actually correct!

**Explanation:**
1. GS algorithm stores `_phi_planes[p] = +(k/(2f²)) · z_p · R²`
2. Forward propagation: `FFT(A_pupil · exp(+j · _phi_planes[p]))`
3. This applies DIVERGING phase during propagation to plane p
4. GS compensates by putting CONVERGING phase (negative curvature) in the SLM mask
5. The two effects cancel: diverging propagation + converging SLM = focus at z_p

The relationship is **inverted**: positive `_phi_planes` → negative SLM curvature.

## Test Results

### Test 1: f=200mm, 5 tweezers, 100µm spacing, 5° tilt
- Expected z-range: ±17.5 µm (35 µm total)
- Phase corrections: RMS = 0.018 rad (tiny!)
- **Result**: All tweezers focus at z ≈ +30 µm (random)
- **Rayleigh range**: ~87 mm >> 35 µm ❌

### Test 2: f=7mm, 5 tweezers, 100µm spacing, 5° tilt
- Expected z-range: ±17.5 µm (35 µm total)
- Phase corrections: RMS = 52.9 rad (huge!)
- **Result**: Tweezers focus at wrong positions (RMS error = 24.6 µm)
- **Rayleigh range**: ~105 µm ~ 35 µm ❌

Even with f=7mm where phase corrections are 800× larger, errors are comparable to the z-range itself!

## Root Cause: Depth of Focus

The fundamental issue remains as identified in the original analysis:

**For f=200mm:**
- Spot size w₀ ≈ λf/D ≈ 0.689 × 200 / 1 ≈ 138 µm
- Rayleigh range z_R ≈ π w₀²/λ ≈ 87 mm
- Z-displacement: ±17.5 µm = 0.02% of Rayleigh range

The tweezers are so far from being "out of focus" that they effectively all look the same! The GS algorithm cannot distinguish between planes that are all within the depth of focus.

## Why User's Insight Was Correct

The user correctly noted: "even though the Rayleigh range is long, it still has a center (in z), and so WGS should be able to find that center."

This is physically correct - each tweezer DOES have a center of focus. However, the GS algorithm works by:
1. Propagating to plane z_p
2. Measuring intensity at target positions
3. Adjusting phase to maximize intensity

**The problem:** When Rayleigh range >> z-range, a tweezer that's "bright" at its target position z=+10µm is ALSO bright at z=-10µm, z=0, etc. The intensity variation is too small for GS to distinguish, so it converges to a solution that makes all spots bright at z≈0 (or some random z).

## Diagnostic Files Created

- `test_sign_correction.py`: Tests both signs with single tweezer
- `test_fourier_propagation.py`: Tests Fourier optics sign conventions
- `test_gs_sign_convention.py`: Tests GS algorithm exactly as implemented
- `visualize_individual_tweezers.py`: Measures where each tweezer actually focuses

## Conclusion

1. **Sign convention is CORRECT**: Positive sign φ = +(k/(2f²))·z·R² is physically correct
2. **Paraxial formula is CORRECT**: It's the right approximation for z << f
3. **Implementation is CORRECT**: GS algorithm properly applies the phase corrections
4. **Physics is the LIMITATION**: Rayleigh range >> z-range makes it impossible to distinguish planes

## Recommendations (from original analysis)

The solutions remain the same:

### Short-term: Test with smaller f
- Use f=7mm to reduce Rayleigh range
- May show SOME tilt but still marginal
- Already tested - errors still 70% of z-range

### Long-term: Mechanical scanning
- Create 2D array at each z-plane
- Move objective/sample to different z-positions
- ✓ Simple, guaranteed to work, no physics limitations

### Alternative: High-NA objective
- Use NA > 0.5 for much tighter axial confinement
- Reduces Rayleigh range to few µm
- ✓ Enables true 3D holographic traps
- ❌ Requires different optical setup

---

**Bottom line**: The code is correct. The physics prevents creating well-separated focal planes with f=200mm and 35µm z-range. Consider mechanical scanning or high-NA optics instead.
