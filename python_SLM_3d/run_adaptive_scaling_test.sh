#!/bin/bash
# Run adaptive GS tests with increasing grid sizes: 5x5, 10x10, 20x20

set -e  # Exit on error

echo "======================================================================"
echo "ADAPTIVE GS SCALING TEST"
echo "======================================================================"
echo ""
echo "This script will run three tests sequentially:"
echo "  1. 5x5 grid (25 tweezers)"
echo "  2. 10x10 grid (100 tweezers)"
echo "  3. 20x20 grid (400 tweezers)"
echo ""
echo "======================================================================"

# Store original test_adaptive_gs.py
ORIGINAL_FILE="test_adaptive_gs.py"
BACKUP_FILE="test_adaptive_gs_backup.py"

echo ""
echo "Creating backup of $ORIGINAL_FILE..."
cp "$ORIGINAL_FILE" "$BACKUP_FILE"

# Function to restore original file on exit
cleanup() {
    echo ""
    echo "Restoring original $ORIGINAL_FILE..."
    mv "$BACKUP_FILE" "$ORIGINAL_FILE"
}
trap cleanup EXIT

# ========== TEST 1: 5x5 Grid ==========
echo ""
echo "======================================================================"
echo "TEST 1/3: Running 5x5 grid (25 tweezers)"
echo "======================================================================"

# Modify parameters for 5x5
sed -i.tmp 's/N_HORIZ = [0-9]*/N_HORIZ = 5/' "$ORIGINAL_FILE"
sed -i.tmp 's/N_VERT = [0-9]*/N_VERT = 5/' "$ORIGINAL_FILE"
sed -i.tmp 's/ITERATIONS = [0-9]*/ITERATIONS = 100/' "$ORIGINAL_FILE"
sed -i.tmp 's/Z_SCAN_RANGE_UM = [0-9.]*$/Z_SCAN_RANGE_UM = 50.0/' "$ORIGINAL_FILE"
sed -i.tmp 's/Z_SCAN_STEPS = [0-9]*/Z_SCAN_STEPS = 11/' "$ORIGINAL_FILE"
sed -i.tmp 's/Z_SCAN_EVERY = [0-9]*/Z_SCAN_EVERY = 5/' "$ORIGINAL_FILE"
rm -f "${ORIGINAL_FILE}.tmp"

echo "Parameters: 5x5, 100 iterations, Z_SCAN_RANGE=50µm, Z_SCAN_STEPS=11"
python "$ORIGINAL_FILE"

# ========== TEST 2: 10x10 Grid ==========
echo ""
echo "======================================================================"
echo "TEST 2/3: Running 10x10 grid (100 tweezers)"
echo "======================================================================"

# Modify parameters for 10x10
sed -i.tmp 's/N_HORIZ = [0-9]*/N_HORIZ = 10/' "$ORIGINAL_FILE"
sed -i.tmp 's/N_VERT = [0-9]*/N_VERT = 10/' "$ORIGINAL_FILE"
sed -i.tmp 's/ITERATIONS = [0-9]*/ITERATIONS = 200/' "$ORIGINAL_FILE"
sed -i.tmp 's/Z_SCAN_RANGE_UM = [0-9.]*$/Z_SCAN_RANGE_UM = 75.0/' "$ORIGINAL_FILE"
sed -i.tmp 's/Z_SCAN_STEPS = [0-9]*/Z_SCAN_STEPS = 15/' "$ORIGINAL_FILE"
sed -i.tmp 's/Z_SCAN_EVERY = [0-9]*/Z_SCAN_EVERY = 5/' "$ORIGINAL_FILE"
rm -f "${ORIGINAL_FILE}.tmp"

echo "Parameters: 10x10, 200 iterations, Z_SCAN_RANGE=75µm, Z_SCAN_STEPS=15"
python "$ORIGINAL_FILE"

# ========== TEST 3: 20x20 Grid ==========
echo ""
echo "======================================================================"
echo "TEST 3/3: Running 20x20 grid (400 tweezers)"
echo "======================================================================"

# Modify parameters for 20x20
sed -i.tmp 's/N_HORIZ = [0-9]*/N_HORIZ = 20/' "$ORIGINAL_FILE"
sed -i.tmp 's/N_VERT = [0-9]*/N_VERT = 20/' "$ORIGINAL_FILE"
sed -i.tmp 's/ITERATIONS = [0-9]*/ITERATIONS = 500/' "$ORIGINAL_FILE"
sed -i.tmp 's/Z_SCAN_RANGE_UM = [0-9.]*$/Z_SCAN_RANGE_UM = 100.0/' "$ORIGINAL_FILE"
sed -i.tmp 's/Z_SCAN_STEPS = [0-9]*/Z_SCAN_STEPS = 21/' "$ORIGINAL_FILE"
sed -i.tmp 's/Z_SCAN_EVERY = [0-9]*/Z_SCAN_EVERY = 10/' "$ORIGINAL_FILE"
rm -f "${ORIGINAL_FILE}.tmp"

echo "Parameters: 20x20, 500 iterations, Z_SCAN_RANGE=100µm, Z_SCAN_STEPS=21, Z_SCAN_EVERY=10"
python "$ORIGINAL_FILE"

# ========== SUMMARY ==========
echo ""
echo "======================================================================"
echo "ALL TESTS COMPLETED!"
echo "======================================================================"
echo ""
echo "Results saved in: slm_output_paraxial/adaptive_test/"
echo ""
echo "To view results, check the following files:"
echo "  - *_adaptive_5x5_*.pkl and corresponding *_xz_profiles.png, *_xy_grid.png"
echo "  - *_adaptive_10x10_*.pkl and corresponding *_xz_profiles.png, *_xy_grid.png"
echo "  - *_adaptive_20x20_*.pkl and corresponding *_xz_profiles.png, *_xy_grid.png"
echo ""
echo "======================================================================"
