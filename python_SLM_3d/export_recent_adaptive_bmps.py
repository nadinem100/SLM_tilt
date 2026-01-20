"""
Export BMP files with blazed grating for the 6 most recent adaptive test results.
"""

import pickle
import numpy as np
from pathlib import Path
from PIL import Image


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


def main():
    print("="*70)
    print("EXPORT BLAZED BMPs FOR RECENT ADAPTIVE TESTS")
    print("="*70)

    # Find all pickle files in adaptive_test directory
    pkl_dir = Path("slm_output_paraxial/adaptive_test")

    if not pkl_dir.exists():
        print(f"\nError: Directory not found: {pkl_dir}")
        return

    pkl_files = sorted(pkl_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)

    if len(pkl_files) == 0:
        print(f"\nNo pickle files found in {pkl_dir}")
        return

    # Take 6 most recent
    n_files = min(6, len(pkl_files))
    pkl_files = pkl_files[:n_files]

    print(f"\nFound {len(pkl_files)} recent pickle files:")
    for i, pkl_path in enumerate(pkl_files, 1):
        print(f"  {i}. {pkl_path.name}")

    # Blazed grating parameters (matching final_run_visualize_paraxial.py)
    fx, fy = 1.0 / 7.0, 0.0

    print(f"\nProcessing files...")

    for i, pkl_path in enumerate(pkl_files, 1):
        print(f"\n[{i}/{n_files}] {pkl_path.name}")

        try:
            # Load pickle
            with open(pkl_path, 'rb') as f:
                bundle = pickle.load(f)

            phase_mask = bundle.phase_mask
            print(f"  Phase mask shape: {phase_mask.shape}")

            # Add blazed grating
            phase_blazed = add_blazed_grating(phase_mask, fx=fx, fy=fy)

            # Output path: same directory, add "_blazepd7.bmp" suffix
            stem = pkl_path.stem  # Filename without extension
            out_bmp = pkl_path.parent / f"{stem}_blazepd7.bmp"

            # Save BMP
            save_phase_bmp(phase_blazed, out_bmp)
            print(f"  ✓ Saved: {out_bmp.name}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
