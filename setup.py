# read env vars and change ds config
import os
import json

config = json.load(open("/workspace/ds_config.json"))
config["train_micro_batch_size_per_gpu"] = int(os.environ.get("BATCH_SIZE_PER_GPU", 100))
config["gradient_accumulation_steps"] = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 100))
config["train_batch_size"] = config["train_micro_batch_size_per_gpu"] * config["gradient_accumulation_steps"]
config["steps_per_print"] = int(os.environ.get("STEPS_PER_PRINT", 100))
json.dump(config, open("/workspace/ds_config.json", "w"))
