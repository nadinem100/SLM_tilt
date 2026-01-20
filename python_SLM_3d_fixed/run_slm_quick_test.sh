#!/bin/bash
#SBATCH --job-name=slm_quick_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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
echo "QUICK TEST - Minimal configuration"
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

# MINIMAL TEST PARAMETERS
N_Z_PLANES=1       # Single plane
ITERATIONS=5       # Just 5 iterations
SCAL=2             # Lower resolution for speed
WAIST_COEFF=9.0    # Standard waist

# Run the script
echo "Starting QUICK TEST SLM simulation..."
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
echo "Quick test completed!"
echo "=========================================="
