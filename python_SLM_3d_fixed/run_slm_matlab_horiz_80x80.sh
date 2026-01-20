#!/bin/bash
#SBATCH --job-name=slm_matlab_80x80
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

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
mkdir -p $REPO_ROOT/slurm_logs

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "MATLAB-style horizontal spacing - 80x80"
echo "=========================================="

module purge
module load python/3.11
module load cuda/12.1

source $REPO_ROOT/.venv/bin/activate
cd $SCRIPT_DIR

echo "Python version:"
python --version
echo "GPU info:"
nvidia-smi
echo ""

# Parameters: 80x80, MATLAB horizontal spacing, -13deg tilt
N_Z_PLANES=10
ITERATIONS=100
SCAL=4
WAIST_COEFF=9.0
TILT_ANGLE_X=-13

echo "Starting SLM simulation..."
echo "Grid: 80x80, MATLAB horizontal spacing, Tilt: ${TILT_ANGLE_X}deg"
echo "N_Z_PLANES=$N_Z_PLANES, ITERATIONS=$ITERATIONS, SCAL=$SCAL, WAIST=$WAIST_COEFF"
python -u test_adaptive_gs_matlab_horiz_80x80.py $N_Z_PLANES $ITERATIONS $SCAL $WAIST_COEFF $TILT_ANGLE_X

deactivate
echo "Job completed!"
