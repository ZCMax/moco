srun -p mm_det \
    --job-name=test_dataset \
    --quotatype=auto \
    --gres=gpu:1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=5 \
    --kill-on-bad-exit=1 \
    python -u test_dataset.py 
