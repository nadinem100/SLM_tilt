"""
Test the memory fix - should show memory going DOWN after cleanup
"""

import numpy as np
import psutil
import gc
import os

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print("="*60)
print("MEMORY FIX VERIFICATION TEST")
print("="*60)

baseline = get_memory_mb()
print(f"Baseline: {baseline:.1f} MB")

# 1. Create large arrays
print("\n1. Creating large arrays...")
big_array_1 = np.random.random((2000, 2000)).astype(np.float32)
big_array_2 = np.random.random((2000, 2000)).astype(np.float32)
after_create = get_memory_mb()
print(f"   Memory: {after_create:.1f} MB (+{after_create-baseline:.1f} MB)")

# 2. OLD cleanup method (doesn't work well)
print("\n2. OLD cleanup (just gc.collect)...")
del big_array_1
gc.collect()
after_old_cleanup = get_memory_mb()
print(f"   Memory: {after_old_cleanup:.1f} MB (freed {after_create-after_old_cleanup:.1f} MB)")

# 3. NEW cleanup method (aggressive)
print("\n3. NEW cleanup (aggressive)...")
del big_array_2

# Close any matplotlib figures
try:
    import matplotlib.pyplot as plt
    plt.close('all')
except:
    pass

# Multiple GC passes
for _ in range(3):
    gc.collect()

after_new_cleanup = get_memory_mb()
print(f"   Memory: {after_new_cleanup:.1f} MB (freed {after_old_cleanup-after_new_cleanup:.1f} MB)")

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Baseline:        {baseline:.1f} MB")
print(f"After arrays:    {after_create:.1f} MB  (+{after_create-baseline:.1f} MB)")
print(f"After OLD clean: {after_old_cleanup:.1f} MB  (freed {after_create-after_old_cleanup:.1f} MB)")
print(f"After NEW clean: {after_new_cleanup:.1f} MB  (freed {after_old_cleanup-after_new_cleanup:.1f} MB)")
print(f"\nTotal freed:     {after_create-after_new_cleanup:.1f} MB")
print(f"Memory overhead: {after_new_cleanup-baseline:.1f} MB")

if (after_new_cleanup - baseline) < 50:
    print("\n✅ PASS: Memory returned to near baseline!")
else:
    print("\n⚠️  WARNING: Still some memory not freed")

print("="*60)