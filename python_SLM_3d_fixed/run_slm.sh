#!/bin/bash
#SBATCH --job-name=slm_20x20_sp30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm_logs/slurm_%j.out
#SBATCH --error=../slurm_logs/slurm_%j.err
#SBATCH --mail-user=nmeister@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Get the directory where sbatch was called from
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Create log directory if it doesn't exist
mkdir -p $REPO_ROOT/slurm_logs

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Repo root: $REPO_ROOT"
echo "Script directory: $SCRIPT_DIR"
echo "=========================================="
echo ""

# Load required modules
module purge
module load python/3.11  # Adjust version as needed
module load cuda/12.1    # Adjust CUDA version as needed

# Activate virtual environment (at repo root)
source $REPO_ROOT/.venv/bin/activate

# Navigate to python_SLM_3d_fixed directory
cd $SCRIPT_DIR

# Print Python and package info
echo "Python version:"
python --version
echo ""
echo "GPU info:"
nvidia-smi
echo ""

# Set parameters (modify these values as needed)
# Grid: 20x20 with 30um spacing (set in test_adaptive_gs.py)
N_Z_PLANES=10
ITERATIONS=100
SCAL=4  # Resolution scaling factor (2 = fast, 4 = accurate)
WAIST_COEFF=9.0  # Beam waist coefficient (9.0 = working config from 2025-11-26)
# Note: Tilt angle is set in test_adaptive_gs.py to -11 degrees (was -13)

# Run the script
echo "Starting SLM simulation (FIXED CORRECTION)..."
echo "Grid: 20x20, Spacing: 30um"
echo "N_Z_PLANES = $N_Z_PLANES"
echo "ITERATIONS = $ITERATIONS"
echo "SCAL = $SCAL"
echo "WAIST_COEFF = $WAIST_COEFF"
python -u test_adaptive_gs.py $N_Z_PLANES $ITERATIONS $SCAL $WAIST_COEFF

# Deactivate virtual environment
deactivate

echo ""
echo "=========================================="
echo "Job completed!"
echo "=========================================="