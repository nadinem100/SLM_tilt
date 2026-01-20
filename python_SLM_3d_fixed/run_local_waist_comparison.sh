#!/bin/bash

# Simple script to run waist coefficient comparison locally
# No SLURM headers - just runs on your Mac

echo "=========================================="
echo "WAIST COEFFICIENT COMPARISON"
echo "=========================================="
echo ""

# Set common parameters
ITERATIONS=150
SCAL=4

# ========== RUN 1: N_Z_PLANES=5, WAIST_COEFF=2.6 ==========
echo "=========================================="
echo "RUN 1/4: N_Z_PLANES=5, WAIST_COEFF=2.6"
echo "=========================================="
python test_adaptive_gs.py 5 $ITERATIONS $SCAL 2.6

echo ""
echo "✓ Completed run 1/4"
echo ""

# ========== RUN 2: N_Z_PLANES=5, WAIST_COEFF=4.0 ==========
echo "=========================================="
echo "RUN 2/4: N_Z_PLANES=5, WAIST_COEFF=4.0"
echo "=========================================="
python test_adaptive_gs.py 5 $ITERATIONS $SCAL 4.0

echo ""
echo "✓ Completed run 2/4"
echo ""

# ========== RUN 3: N_Z_PLANES=10, WAIST_COEFF=2.6 ==========
echo "=========================================="
echo "RUN 3/4: N_Z_PLANES=10, WAIST_COEFF=2.6"
echo "=========================================="
python test_adaptive_gs.py 10 $ITERATIONS $SCAL 2.6

echo ""
echo "✓ Completed run 3/4"
echo ""

# ========== RUN 4: N_Z_PLANES=10, WAIST_COEFF=4.0 ==========
echo "=========================================="
echo "RUN 4/4: N_Z_PLANES=10, WAIST_COEFF=4.0"
echo "=========================================="
python test_adaptive_gs.py 10 $ITERATIONS $SCAL 4.0

echo ""
echo "✓ Completed run 4/4"
echo ""

echo ""
echo "=========================================="
echo "ALL RUNS COMPLETED!"
echo "=========================================="
echo "Results saved with different parameters:"
echo "  - 5planes_waist2.6"
echo "  - 5planes_waist4.0"
echo "  - 10planes_waist2.6"
echo "  - 10planes_waist4.0"
echo "=========================================="
