#!/bin/bash
#SBATCH --job-name=slm_80x80_sp7.2
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

# Set parameters for 80x80 grid with 7.2um spacing
N_Z_PLANES=10
ITERATIONS=100
SCAL=4  # Resolution scaling factor (2 = fast, 4 = accurate)
WAIST_COEFF=9.0  # Beam waist coefficient (9.0 = working config)
TILT_ANGLE=-11  # Tilt angle in degrees (changed from -13)

# Grid parameters (hardcoded in test script)
N_HORIZ=80
N_VERT=80
SPACING_UM=7.2

# Run the script
echo "Starting SLM simulation (FIXED CORRECTION)..."
echo "Grid: ${N_HORIZ}x${N_VERT}, Spacing: ${SPACING_UM}um"
echo "N_Z_PLANES = $N_Z_PLANES"
echo "ITERATIONS = $ITERATIONS"
echo "SCAL = $SCAL"
echo "WAIST_COEFF = $WAIST_COEFF"
echo "TILT_ANGLE = $TILT_ANGLE"
python -u test_adaptive_gs_80x80.py $N_Z_PLANES $ITERATIONS $SCAL $WAIST_COEFF $TILT_ANGLE

# Deactivate virtual environment
deactivate

echo ""
echo "=========================================="
echo "Job completed!"
echo "=========================================="
