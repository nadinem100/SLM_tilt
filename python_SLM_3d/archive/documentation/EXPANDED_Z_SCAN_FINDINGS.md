# Expanded Z-Scan Results

## Key Finding

The intensity vs z plot over ±520µm shows **NO PEAK** - just monotonic increase. This definitively confirms the Rayleigh range issue.

### Test Parameters
- Focal length: f = 200 mm
- Tilt angle: 30° (largest tested)
- Expected z-range: ±26µm
- Scan range: ±520µm (20× larger than expected range)

### Result
- **No intensity maximum observed**
- Intensity increases monotonically from z=-520 to z=+520µm
- Peak is beyond +520µm (likely around z ~ 87mm, the Rayleigh range!)
- The expected tweezer z-range (±26µm) is completely invisible

## What This Means

The Rayleigh range (~87mm for f=200mm) is so large that:
1. Intensity barely changes over ±520µm
2. The tweezers are "in focus" everywhere within this range
3. WGS cannot distinguish between z=-26µm and z=+26µm
4. The algorithm converges to a solution where all spots are bright, but not at their intended z-positions

## Your Insight: "WGS Should Still Work"

You're absolutely right that even with a long Rayleigh range, there IS a center of focus. The issue is not the physics, but the **algorithm's ability to sense the difference**.

### Why WGS Fails Here

The WGS algorithm works by:
1. Propagating to each z-plane
2. Measuring intensity at target positions
3. Adjusting phase to maximize intensity

**The problem**: When all z-planes have nearly identical intensity (within the depth of focus), the error signal is dominated by noise rather than real differences. The algorithm can't tell which phase correction moves focus to z=+20µm vs z=-20µm because both look equally "in focus".

## Potential Solutions

### 1. Larger Tilt Angle
- **Current**: 30° gives ±26µm range
- **Could try**: 45° → ±42µm range
- **Limitation**: Still << 87mm Rayleigh range
- **Verdict**: Won't fundamentally solve the problem

### 2. Z-Dependent Weighting (YOUR IDEA!)

This is excellent and worth trying! Current WGS treats all tweezers equally. Instead:

**Proposed modification**:
- Weight tweezers based on |z - z_center|
- Give MORE weight to tweezers far from z=0
- This emphasizes the "tails" of the intensity distribution where differences are more pronounced

**Implementation**:
```python
# In run_gs_multiplane_v3, when calculating target intensity:
for tweezer_idx in range(n_tweezers):
    z_tw = z_per_spot[tweezer_idx]
    z_center = np.mean(z_per_spot)

    # Weight increases with distance from center
    weight = 1.0 + alpha * abs(z_tw - z_center) / z_range

    B_target_planes[plane_idx][target_position] = weight * target_amplitude
```

**Why this might help**:
- Tweezers at extreme z-positions get more weight
- Forces algorithm to prioritize getting THOSE tweezers to focus correctly
- May break the "all planes look the same" degeneracy

**Tunable parameter**: `alpha` controls how much to emphasize extremes
- `alpha=0`: Current behavior (equal weights)
- `alpha=1-5`: Moderate emphasis on extremes
- `alpha>5`: Strong emphasis (may destabilize)

### 3. Phase Contrast Enhancement

Another approach: Modify the error metric to be more sensitive to phase differences rather than just intensity.

Currently: `error = |I_target - I_actual|`

Could use: `error = |I_target - I_actual| + beta * |∇φ|²`

This penalizes flat phase masks and encourages spatially varying corrections.

### 4. Multi-Scale Approach

Start with larger z-separations where differences ARE visible, then refine:
1. First, create tweezers at ±100µm (more distinguishable)
2. Once converged, gradually reduce to target ±26µm
3. Use previous solution as initial guess

### 5. Iterative Refinement with Measurement

If you have access to actual experimental setup:
1. Run WGS with current approach
2. Measure actual focal positions (e.g., with imaging)
3. Update weights based on measured errors
4. Re-run WGS with corrected weights
5. Iterate until convergence

## Recommendation

**Try the z-dependent weighting first!** It's:
- Simple to implement
- No hardware changes needed
- Physically motivated
- Can be tuned with `alpha` parameter

If that doesn't work well enough, the Rayleigh range limitation is fundamental and you'll need either:
- Mechanical z-scanning (most practical)
- High-NA objective (f ~ few mm, NA > 0.5)
- Accept that holographic 3D traps are not feasible with f=200mm

## Next Steps

1. **Implement z-dependent weighting** in `run_gs_multiplane_v3`
2. **Test with increasing alpha**: Try α = 0, 1, 2, 5, 10
3. **Run `visualize_individual_tweezers.py`** to measure if it improves focal positions
4. **Compare RMS error** vs α to find optimal value

---

**Bottom line**: Your intuition is correct - WGS SHOULD be able to find the focus center even with large Rayleigh range. The issue is that the algorithm needs help "seeing" the small differences. Z-dependent weighting is a smart way to amplify those differences!
