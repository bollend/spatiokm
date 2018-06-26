#!/bin/sh
#SBATCH --job-name=IR2nc
#SBATCH --account=dylanb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#   (use the normal partition=default)
#SBATCH --output=/home/dylanb/astronomy/MCMC_desktop/MCMC_objects/IRAS19135+3937_var2/results_nested_cos/stdout.log
#SBATCH --error=/home/dylanb/astronomy/MCMC_desktop/MCMC_objects/IRAS19135+3937_var2/results_nested_cos/stderr.log

mpirun -np 24 python /home/dylanb/astronomy/MCMC_desktop/MCMC_general_nested_cos/syn_dynspec_MCMC.py -object IRAS19135+3937_var2 -line halpha_abs
