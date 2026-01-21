#!/bin/bash
#SBATCH --job-name=slm_60x60_matlab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=20:00:00
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

# Copy the matlab spacing file and modify parameters
cp test_adaptive_gs_matlab_spacing.py test_adaptive_gs_temp.py

# Modify parameters using sed
sed -i 's/N_HORIZ = .*/N_HORIZ = 60/' test_adaptive_gs_temp.py
sed -i 's/N_VERT = .*/N_VERT = 60/' test_adaptive_gs_temp.py
sed -i 's/TILT_ANGLE_X = .*/TILT_ANGLE_X = -13  # degrees/' test_adaptive_gs_temp.py
sed -i 's/N_Z_PLANES = .*/N_Z_PLANES = 5/' test_adaptive_gs_temp.py
sed -i 's/ITERATIONS = .*/ITERATIONS = 100/' test_adaptive_gs_temp.py
sed -i 's/SCAL = .*/SCAL = 4/' test_adaptive_gs_temp.py
sed -i 's/WAIST_UM = .*/WAIST_UM = 9 \/ 2 * 1e3  # microns/' test_adaptive_gs_temp.py

echo "Starting SLM simulation..."
echo "Grid: 60x60, MATLAB spacing, Tilt: -13deg"
echo "N_Z_PLANES=5, ITERATIONS=100, SCAL=4, WAIST=9.0"
python -u test_adaptive_gs_temp.py

# Clean up temp file
rm test_adaptive_gs_temp.py

deactivate
echo "Job completed!"
