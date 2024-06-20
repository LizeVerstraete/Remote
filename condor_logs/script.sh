#!/usr/bin/bash
echo "Starting Job"
export PATH="/esat/biomeddata/kkontras/r0786880/anaconda3/bin:$PATH"
source /esat/biomeddata/kkontras/r0786880/anaconda3/etc/profile.d/conda.sh
conda activate /esat/biomeddata/kkontras/r0786880/anaconda3/envs/remote_env

which python
python -V
echo "Current PATH: $PATH"
cd /esat/biomeddata/kkontras/r0786880/models/remote_new
export PYTHONPATH='/esat/biomeddata/kkontras/r0786880/models/remote_new/:$PYTHONPATH'
echo $PWD
num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ $num_gpus -eq 1 ]; then
    echo "We have 1 GPU"
else
    echo "We have $num_gpus GPUs"
fi
echo "Current PATH: $PATH"
python /esat/biomeddata/kkontras/r0786880/models/remote_new/preprocessing_parallel.py
echo "Job finished"