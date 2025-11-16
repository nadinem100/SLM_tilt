#!/bin/bash
# Run adaptive GS tests with 20x20 grid at different tilt angles

set -e  # Exit on error

echo "======================================================================"
echo "ADAPTIVE GS N_Z_PLANES TEST"
echo "======================================================================"
echo ""
echo "This script will run tests with current parameters but varying N_Z_PLANES:"
echo "  Test 1: N_Z_PLANES = 10"
echo "  Test 2: N_Z_PLANES = 20"
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

# Array of N_Z_PLANES values to test
N_Z_PLANES_ARRAY=(10 20)

# Loop through each N_Z_PLANES value
for i in "${!N_Z_PLANES_ARRAY[@]}"; do
    N_PLANES="${N_Z_PLANES_ARRAY[$i]}"
    TEST_NUM=$((i + 1))

    echo ""
    echo "======================================================================"
    echo "TEST $TEST_NUM/2: Running with N_Z_PLANES = ${N_PLANES}"
    echo "======================================================================"

    # Modify N_Z_PLANES parameter
    sed -i.tmp "s/N_Z_PLANES = [0-9]*/N_Z_PLANES = ${N_PLANES}/" "$ORIGINAL_FILE"
    rm -f "${ORIGINAL_FILE}.tmp"

    echo "Parameters: N_Z_PLANES=${N_PLANES} (keeping all other parameters from test_adaptive_gs.py)"
    python "$ORIGINAL_FILE"
done

# ========== SUMMARY ==========
echo ""
echo "======================================================================"
echo "ALL TESTS COMPLETED!"
echo "======================================================================"
echo ""
echo "Results saved in: slm_output_paraxial/adaptive_test/"
echo ""
echo "To view results, check the following files:"
echo "  - Files with N_Z_PLANES variations"
echo "  - Corresponding .pkl files, _blazepd7.bmp files, and diagnostic images"
echo ""
echo "======================================================================"
