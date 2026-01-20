#!/bin/bash
# Setup script for Mac GPU environment (Apple Silicon with MPS)
# Run this once to create and configure your environment

set -e  # Exit on error

echo "=========================================="
echo "Mac GPU Environment Setup (Apple Silicon)"
echo "=========================================="

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "WARNING: This script is optimized for Apple Silicon (M1/M2/M3/M4)."
    echo "Your system: $(uname -m)"
fi

# 1. Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv_mac_gpu

# 2. Activate it
echo "Activating virtual environment..."
source .venv_mac_gpu/bin/activate

# 3. Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# 4. Install PyTorch with MPS support
echo ""
echo "Installing PyTorch with MPS support for Apple Silicon..."
pip install torch torchvision torchaudio

# 5. Install other dependencies
echo ""
echo "Installing Python packages..."
pip install numpy scipy pyyaml pillow matplotlib

# 6. Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'✓ SciPy: {scipy.__version__}')"
python -c "import yaml; print('✓ PyYAML installed')"
python -c "from PIL import Image; print('✓ Pillow installed')"
python -c "import matplotlib; print(f'✓ Matplotlib: {matplotlib.__version__}')"

# Test PyTorch and MPS
echo ""
echo "Testing PyTorch GPU support..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'  MPS available: {torch.backends.mps.is_available()}'); print(f'  MPS built: {torch.backends.mps.is_built()}')"

if python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" 2>/dev/null; then
    echo "  ✓ MPS GPU support is available!"
else
    echo "  ⚠ MPS not available - will fall back to CPU"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate this environment in the future:"
echo "  source .venv_mac_gpu/bin/activate"
echo ""
echo "To run your script:"
echo "  cd python_SLM_3d_fixed"
echo "  python test_adaptive_gs_matlab_spacing.py"
echo ""