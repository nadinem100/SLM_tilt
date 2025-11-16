#!/bin/bash
#SBATCH --job-name=slm_adaptive_gs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err

# Create log directory if it doesn't exist
mkdir -p $SLURM_SUBMIT_DIR/slurm_logs

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "=========================================="
echo ""

# Load required modules
module purge
module load python/3.11  # Adjust version as needed
module load cuda/12.1    # Adjust CUDA version as needed

# Activate virtual environment (use absolute path)
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Navigate to script directory
cd $SLURM_SUBMIT_DIR/python_SLM_3d

# Print Python and package info
echo "Python version:"
python --version
echo ""
echo "GPU info:"
nvidia-smi
echo ""

# Run the script
echo "Starting SLM simulation..."
python test_adaptive_gs.py

# Deactivate virtual environment
deactivate

echo ""
echo "=========================================="
echo "Job completed!"
echo "=========================================="
