# Tilted Plane Issue - Root Cause Analysis

## Problem Statement
The WGS algorithm successfully creates tweezer arrays in a single plane, but when attempting to create tweezers on a tilted plane (5° tilt, z-range ±25 µm), all tweezers cluster near z=0 instead of following the tilt.

## What We've Confirmed

### ✅ Working Correctly:
1. **Paraxial formula is correct**: φ = (k/2f²)·z·R²
2. **Phase corrections are calculated**: Non-zero RMS values (0.065 rad for f=200mm, 52 rad for f=7mm)
3. **Tweezers assigned to planes**: Each tweezer is correctly assigned to one of 5 z-planes
4. **Algorithm runs**: GS converges with reasonable error signals

### ❌ Not Working:
1. **Final focal positions**: Tweezers do NOT focus at their assigned z-positions
2. **No visible tilt**: X-Z plots show tweezers clustered near z=0, not following the magenta tilt line
3. **Large errors**: Measured focal positions have RMS error of 24-36 µm (larger than the 35 µm z-range!)

## Root Cause: Depth of Focus

The fundamental issue is that the **Rayleigh range (depth of focus) is much larger than the z-displacements**.

### Rayleigh Range Calculation
For a Gaussian beam: z_R = π·w₀²/λ

For f=200mm with a ~1mm pupil diameter:
- Spot size w₀ ≈ λ·f/D ≈ 0.689 µm × 200 mm / 1 mm ≈ 138 µm
- Rayleigh range z_R ≈ π·(138)²/0.689 ≈ **87,000 µm** = 87 mm!

For f=7mm:
- Spot size w₀ ≈ 0.689 µm × 7 mm / 1 mm ≈ 4.8 µm
- Rayleigh range z_R ≈ π·(4.8)²/0.689 ≈ **105 µm**

### The Problem
Your z-displacements are ±25 µm (50 µm total range). For f=200mm, this is only **0.03% of the Rayleigh range**. The tweezers are so far from being "out of focus" that they effectively all look the same!

Even for f=7mm, 50 µm is only ~50% of the Rayleigh range, which is marginal.

## Why GS Can't Fix This

The Gerchberg-Saxton algorithm works by:
1. Propagating field to focal plane
2. Replacing amplitude with target, keeping phase
3. Propagating back to SLM plane
4. Keeping phase, replacing amplitude with input beam

For multi-plane GS, it propagates to different z-planes and tries to make each tweezer bright at its assigned z. **But if the depth of focus is huge, a tweezer that's "in focus" at z=-25 µm is ALSO essentially "in focus" at z=0 and z=+25 µm!**

The algorithm can't distinguish between them, so it just makes all tweezers bright everywhere within the Rayleigh range.

## Evidence from Testing

### Test 1: f=200mm, 10° tilt, 5 tweezers
- Phase corrections: RMS = 0.065 rad, max = 0.16 rad
- Expected z-range: ±17.6 µm
- **Result**: All tweezers focus at ±30 µm (edges of scan range) - RANDOM positions
- **Rayleigh range**: 87 mm >> 35 µm ❌

### Test 2: f=7mm, 5° tilt, 5 tweezers
- Phase corrections: RMS = 52.8 rad, max = 131 rad (huge!)
- Expected z-range: ±17.5 µm
- **Result**: Large errors (RMS = 24.8 µm), not following tilt
- **Rayleigh range**: 105 µm ~ 50 µm ❌

Even with f=7mm where the phase corrections are 800× larger, it still doesn't work well!

## Why This Worked for Others

Papers showing tilted optical tweezer planes typically use:

1. **Much tighter focusing**: NA ≈ 1.0 (water immersion), giving w₀ < 1 µm and z_R < 5 µm
2. **Smaller z-range**: Tilts of 1-2° over 100 µm extent → z-range of only 2-4 µm
3. **Or different method**: Axicons/light sheets instead of holographic focusing

Your setup:
- NA ≈ D/(2f) ≈ 2mm/(2×200mm) = 0.005 (very low!)
- 5° tilt over 570 µm extent → z-range = 50 µm
- Rayleigh range = 87 mm >> 50 µm

## Possible Solutions

### Option 1: Reduce Focal Length (Limited Success)
Using f=7mm instead of f=200mm:
- ✓ Increases phase corrections by 800×
- ✓ Reduces Rayleigh range from 87mm to 105µm
- ⚠ Still marginal - may show slight tilt but not clean separation
- ❌ Changes your optical setup significantly

### Option 2: Reduce Tilt Angle / Z-Range
- Use 1° instead of 5° → z-range = 10 µm instead of 50 µm
- Smaller z-range is easier for GS to distinguish
- ❌ May not be useful for your application

### Option 3: Increase NA (High Power Objective)
- Use microscope objective with NA > 0.5
- Spot size w₀ ~ λ/(2·NA) → much tighter focus
- Rayleigh range z_R ~ λ/NA² → much shorter
- ✓ This is how most holographic tweezer papers work
- ❌ Requires different optical setup, smaller field of view

### Option 4: Mechanical Z-Scanning
Instead of trying to create all z-planes simultaneously:
- Create single-plane tweezer array (works great!)
- Move objective or sample to different z-positions
- Capture images at each z-position
- ✓ Simple, guaranteed to work
- ⚠ Not simultaneous (but may be fast enough)

### Option 5: Temporal Multiplexing
- Rapidly switch between different phase masks for different z-planes
- If switching is fast enough (>100 Hz), atoms see time-averaged potential
- ✓ Allows large z-separations
- ⚠ Complex, requires fast SLM

## Recommendation

For your current setup (f=200mm, 5° tilt):

**The tilted plane approach is fundamentally limited by the huge Rayleigh range.**

Your best options are:

1. **Short term**: Test with f=7mm to see if you can get ANY tilt to show up (may work marginally)

2. **Long term**: Use **mechanical z-scanning** - create a 2D array at each z-plane and move the sample/objective. This is simpler and more reliable than trying to fight the physics of depth of focus.

3. **If you need true 3D holographic traps**: Switch to a high-NA objective (NA > 0.5) which gives much tighter axial confinement.

## Files for Reference

- Paraxial formula implementation: `slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial.py` (line 761)
- Diagnostic tests: `test_simple_tilt.py`, `visualize_individual_tweezers.py`
- Latest visualization: `slm_output_paraxial/251106-110853_f200000um_sp30.0um_planes5_tilt_5_tw_20/`

## Physics References

- Rayleigh range: Born & Wolf, "Principles of Optics" Chapter 8
- Holographic optical tweezers: Curtis et al., Opt. Commun. 207, 169 (2002)
- 3D trap arrays: Dufresne & Grier, Rev. Sci. Instrum. 69, 1974 (1998)

---

**Bottom line**: The paraxial formula is correct, but physics limits what you can achieve with f=200mm and 50 µm z-range. The depth of focus is simply too large to create well-separated focal planes.
