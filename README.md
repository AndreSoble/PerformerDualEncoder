# PerformerDualEncoder
In this paper a train funtion will be implemented which will be used to train a performer for language agnostic representations.

### Usage
```python
from transformers import RobertaTokenizer
from modelling_dual_encoder_performer import DualEncoderPerformer

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
model = DualEncoderPerformer(num_tokens=tokenizer.vocab_size)
sentence1 = tokenizer(["Ich bin Andre", "Ich bin Andre"],
                      add_special_tokens=True, return_tensors="pt",
                      padding=True)
sentence2 = tokenizer(["I am Andre", "I need support"],
                           add_special_tokens=True, return_tensors="pt",
                           padding=True)
print(model.get_similarity(sentence1, sentence2))
```

### Training
At first modify the training parameters in env.list. These environ vars will be used during training.
In addition there is a ds_config.json. 
In this json you can modify training parameters like learning rate.
For more information about the ds_config.json check out deepspeed https://www.deepspeed.ai/docs/config-json/

The env.list parameters will be prioritized
####Run a training on docker
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
