# PerformerDualEncoder - Work in Progress
In this paper a train funtion will be implemented which will be used to train a performer for language agnostic representations.

### Install dependencies
```
pip install -r requirements.txt
```

### Usage (A pretrained model has not been released yet but will be in the future)
```python
from transformers import AutoTokenizer
from modelling_dual_encoder_performer import DualEncoderPerformer

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = DualEncoderPerformer(num_tokens=tokenizer.vocab_size)
sentences1 = tokenizer(["Ich bin Andre", "Ich bin Andre"],
                      add_special_tokens=True, return_tensors="pt",
                      padding=True)
sentences2 = tokenizer(["I am Andre", "I need support"],
                           add_special_tokens=True, return_tensors="pt",
                           padding=True)
print(model.get_similarity(sentences1, sentences2))
```
This code should output something like the following:
```
tensor([0.6409, 0.4435])
```

### Training

At first modify the training parameters in env.list. These environ vars will be used during training.
In addition there is a ds_config.json. 
In this json you can modify training parameters like learning rate.
For more information about the ds_config.json check out deepspeed https://www.deepspeed.ai/docs/config-json/
The env.list parameters will be prioritized

#### Run a training on docker

At first build the images by running
```
bash build_images.sh
```
then run the training using:
```
docker run -d --cpuset-cpus="0-17" --runtime=nvidia -it -p 6016:6016 -v /path/to/data/storage:/storage \ 
                                                                     -v /path/to/model_save_dir/results:/results \
                                                                     --env-file ./env.list \
                                                                     --name transformer \
                                                                      performer_job \
```
The training script will automatically download the OPUS-100 Dataset into the storage directory and start training