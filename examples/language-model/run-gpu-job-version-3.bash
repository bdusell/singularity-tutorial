#!/bin/bash

echo "Running on host $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "nvidia-smi output:"
nvidia-smi

singularity exec --nv /scratch365/$USER/version-3.sif pipenv run "$@"
