#!/bin/bash
#SBATCH --job-name=slm_80x80_tilt_pos5
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
echo "Running on node: $SLURM_NODELIST"
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

# Parameters: 80x80, 7.2um, +5 degree tilt (POSITIVE)
N_Z_PLANES=10
ITERATIONS=100
SCAL=4
WAIST_COEFF=9.0
TILT_ANGLE=5  # POSITIVE tilt

echo "Starting SLM simulation..."
echo "Grid: 80x80, Spacing: 7.2um, Tilt: +${TILT_ANGLE}deg"
echo "N_Z_PLANES=$N_Z_PLANES, ITERATIONS=$ITERATIONS, SCAL=$SCAL, WAIST=$WAIST_COEFF"
python -u test_adaptive_gs_80x80.py $N_Z_PLANES $ITERATIONS $SCAL $WAIST_COEFF $TILT_ANGLE

deactivate
echo "Job completed!"
