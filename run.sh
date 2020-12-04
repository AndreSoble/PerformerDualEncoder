python /workspace/setup.py
if [ "$TRAINER" = "deepspeed" ]; then
  deepspeed /workspace/training.py --deepspeed --deepspeed_config /workspace/ds_config.json
else
  python /workspace/training_huggingface.py
fi
python /workspace/evaluation.py