# The flag `-l gpu_card=1` is necessary when using the GPU queue.
qsub \
  -N singularity-tutorial-version-3 \
  -o output-version-3.txt \
  -q gpu \
  -l gpu_card=1 \
  run-gpu-job-version-3.bash \
    python main.py --cuda --epochs 6
