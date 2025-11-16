# SLM Holographic Tweezers - Main Directory

## Main Files

### To Run WGS Algorithm
- **`final_run_visualize_paraxial.py`** - Main script to run WGS and create tweezer patterns
  - Uses paraxial defocus formula (correct for optical tweezers)
  - Saves results to `slm_output_paraxial/`
  - Creates visualizations of the phase mask and intensity patterns

### Class Files
- **`slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial.py`** - Main class (PARAXIAL formula, recommended)
  - Uses φ = +(k/(2f²))·z·R² for defocus corrections
  - Correct for z << f (optical tweezers regime)

- **`slm_tweezers_class_WITH_AUTO_CLEANUP.py`** - Original class (exact formula, not recommended)
  - Uses φ = -k·z·R²/(2f(f-z)) which has problematic z/f scaling
  - Kept for reference only

### Visualization Tools
- **`visualize_from_pickle.py`** - Load saved results and try different visualization strategies
  - No re-running of WGS
  - Tests multiple gamma corrections and intensity scalings

- **`visualize_individual_tweezers.py`** - Measure where each tweezer actually focuses
  - Scans through z and measures intensity at each tweezer position
  - Creates intensity vs z plots
  - Compares measured focal positions to expected positions

## Archive Folders

- **`archive/documentation/`** - Analysis documents and explanations
  - TILTED_PLANE_ISSUE_SUMMARY.md - Root cause analysis (Rayleigh range issue)
  - EXPANDED_Z_SCAN_FINDINGS.md - Results from expanded z-scanning
  - SIGN_INVESTIGATION_RESULTS.md - Sign convention investigation
  - And other explanatory documents

- **`archive/diagnostic_tests/`** - Test scripts used during debugging
  - test_*.py - Various diagnostic tests
  - diagnose_*.py - Phase mask analysis scripts

- **`archive/old_visualizations/`** - Old/deprecated visualization scripts

- **`old_files/`** - Previous versions of class files

## Quick Start

1. **Run WGS algorithm:**
   ```bash
   python final_run_visualize_paraxial.py
   ```

2. **Visualize saved results:**
   ```bash
   python visualize_from_pickle.py
   ```

3. **Analyze focal positions:**
   ```bash
   python visualize_individual_tweezers.py
   ```

## Key Finding

**For f=200mm with ±17.5µm z-range:** The Rayleigh range (~87mm) is much larger than the z-displacement range, making it impossible for WGS to create well-separated focal planes. See `archive/documentation/TILTED_PLANE_ISSUE_SUMMARY.md` for details.

**Recommended solutions:**
- Mechanical z-scanning (most practical)
- High-NA objective (f ~ few mm, NA > 0.5)
- Z-dependent weighting in WGS (experimental, may help marginally)
