python /workspace/setup.py
python /workspace/training_huggingface.py
#deepspeed /workspace/training.py --deepspeed --deepspeed_config /workspace/ds_config.json
python /workspace/evaluation.py