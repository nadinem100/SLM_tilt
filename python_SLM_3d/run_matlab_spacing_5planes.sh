#!/bin/bash
#SBATCH --job-name=matlab_sp_5planes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_logs/matlab_spacing_5planes_%j.out
#SBATCH --error=slurm_logs/matlab_spacing_5planes_%j.err
#SBATCH --mail-user=nmeister@caltech.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Create log directory if it doesn't exist
mkdir -p $SLURM_SUBMIT_DIR/slurm_logs

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Configuration: MATLAB spacing, -13deg tilt, 5 z-planes, 100 iter, tol=1e-3"
echo "=========================================="
echo ""

# Load required modules
module purge
module load python/3.11
module load cuda/12.1

# Activate virtual environment (use absolute path)
source $SLURM_SUBMIT_DIR/../.venv/bin/activate

# Navigate to script directory
cd $SLURM_SUBMIT_DIR

# Print Python and package info
echo "Python version:"
python --version
echo ""
echo "GPU info:"
nvidia-smi
echo ""

# Run the script with 5 z-planes
echo "Starting SLM simulation with MATLAB spacing (5 z-planes)..."
python -u test_adaptive_gs_matlab_spacing_cluster.py --n_planes 5

# Deactivate virtual environment
deactivate

echo ""
echo "=========================================="
echo "Job completed!"
echo "=========================================="
