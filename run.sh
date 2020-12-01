python /workspace/setup.py
deepspeed /workspace/training.py --deepspeed --deepspeed_config /workspace/ds_config.json
python /workspace/evaluate.py