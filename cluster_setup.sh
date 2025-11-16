#!/bin/bash
# Cluster setup script for SLM simulation
# Run this ONCE on the cluster to set up your environment

set -e  # Exit on error

echo "=========================================="
echo "SLM Cluster Environment Setup"
echo "=========================================="

# 1. Load required modules
echo "Loading modules..."
module purge
module load python/3.11  # Adjust version as needed
module load cuda/12.1    # Adjust CUDA version as needed
module load gcc

# 2. Create virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# 3. Activate it
echo "Activating virtual environment..."
source .venv/bin/activate

# 4. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install dependencies
echo "Installing Python packages..."
pip install numpy scipy pyyaml pillow matplotlib

# 6. Install CuPy (GPU acceleration)
echo "Installing CuPy for GPU support..."
# Detect CUDA version and install appropriate CuPy
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "Detected CUDA version: $CUDA_VERSION"

if [[ $CUDA_VERSION == 12.* ]]; then
    pip install cupy-cuda12x
elif [[ $CUDA_VERSION == 11.* ]]; then
    pip install cupy-cuda11x
else
    echo "WARNING: Unknown CUDA version. Installing cupy-cuda12x by default"
    pip install cupy-cuda12x
fi

# 7. Verify installation
echo ""
echo "Verifying installation..."
python -c "import numpy, scipy, yaml, cupy; print('✓ All packages installed successfully')"
python -c "import cupy as cp; print(f'✓ CuPy version: {cp.__version__}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate this environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To submit a job:"
echo "  sbatch run_slm.sh"
