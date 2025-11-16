"""
MINIMAL DEBUG SCRIPT - Find the Memory Bottleneck

This runs a tiny version of your code with detailed memory tracking.
Run this FIRST to see what's causing the slowdown.

Usage:
    python debug_memory_minimal.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gc
import psutil
import os
import sys

# Add project to path (adjust if needed)
sys.path.insert(0, '../slm_parameters.yml')

from slm_tweezers_class_WITH_AUTO_CLEANUP import SLMTweezers

# TINY parameters for fast testing
N_HORIZ = 2  # Just 2x2 grid
N_VERT = 2
SPACING_UM = 15.0
ITERATIONS = 5  # Only 5 iterations
REDSLM = 1
SCAL = 2  # Smaller scaling
WAIST_UM = 9 / 2 * 1e3

YAML_PATH = "../slm_parameters.yml"
OUT_DIR = Path("/tmp/slm_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_memory_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_memory(label):
    """Print memory usage with label"""
    mem = get_memory_mb()
    print(f"  📊 {label}: {mem:.1f} MB")
    return mem


def main():
    print("="*70)
    print("🔍 MEMORY DEBUG - Finding the Bottleneck")
    print("="*70)
    print(f"Running tiny test: {N_HORIZ}x{N_VERT} grid, {ITERATIONS} iterations")
    print("="*70)
    
    mem_start = print_memory("Initial memory")
    
    # Step 1: Setup
    print("\n1️⃣ Setting up SLM...")
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    print_memory("After SLM creation")
    
    slm.init_fields(waist_um=WAIST_UM)
    mem_after_init = print_memory("After init_fields")
    
    slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM, 
                        odd_tw=1, box1=1)
    mem_after_grid = print_memory("After set_target_grid")
    
    slm.set_optics(wavelength_um=0.689, focal_length_um=200000.0)
    mem_after_optics = print_memory("After set_optics")
    
    # Step 2: Assign planes
    print("\n2️⃣ Assigning planes...")
    slm.assign_planes_from_tilt(
        tilt_x_deg=10.0,
        z_min_um=-50.0,
        z_max_um=50.0,
        n_planes=3  # Only 3 planes for testing
    )
    mem_after_planes = print_memory("After assign_planes")
    
    # Step 3: Run GS
    print(f"\n3️⃣ Running GS ({ITERATIONS} iterations)...")
    mem_before_gs = get_memory_mb()
    
    slm.run_gs_multiplane_v3(iterations=ITERATIONS, Gg=0.6, verbose=True)
    
    mem_after_gs = print_memory("After GS complete")
    
    # Step 4: Save
    print("\n4️⃣ Saving results...")
    bundle = slm.save_pickle(out_dir=str(OUT_DIR), label="_debug")
    mem_after_save = print_memory("After save")
    
    # Step 5: Cleanup test
    print("\n5️⃣ Testing cleanup...")
    del slm
    gc.collect()
    mem_after_cleanup = print_memory("After del slm + gc.collect()")
    
    # Summary
    print("\n" + "="*70)
    print("📈 MEMORY SUMMARY")
    print("="*70)
    print(f"Setup phase: +{mem_after_optics - mem_start:.1f} MB")
    print(f"Plane assignment: +{mem_after_planes - mem_after_optics:.1f} MB")
    print(f"GS iterations: +{mem_after_gs - mem_before_gs:.1f} MB  ⚠️ CHECK THIS")
    print(f"Save: +{mem_after_save - mem_after_gs:.1f} MB")
    print(f"Cleanup: {mem_after_cleanup - mem_after_save:.1f} MB")
    print(f"\nTotal memory used: {mem_after_gs - mem_start:.1f} MB")
    print(f"Final memory: {mem_after_cleanup:.1f} MB")
    print("="*70)
    
    # Check for leaks
    if mem_after_cleanup > (mem_start + 50):
        print("\n⚠️  WARNING: Possible memory leak detected!")
        print(f"   Memory didn't return to baseline after cleanup")
        print(f"   Started: {mem_start:.1f} MB, Ended: {mem_after_cleanup:.1f} MB")
    else:
        print("\n✅ Memory cleanup looks good!")
    
    print(f"\n✅ Results saved to: {bundle.file}")


if __name__ == "__main__":
    # Don't create plots in this debug version
    plt.ioff()  # Turn off interactive mode
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Force cleanup
        plt.close('all')
        gc.collect()
        print(f"\n🏁 Final memory: {get_memory_mb():.1f} MB")
