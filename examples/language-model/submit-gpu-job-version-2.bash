# The flag `-l gpu_card=1` is necessary when using the GPU queue.
qsub \
  -N singularity-tutorial-version-2 \
  -o output-version-2.txt \
  -q gpu \
  -l gpu_card=1 \
  run-gpu-job-version-2.bash \
    python3 main.py --cuda --epochs 6
