# import libraries
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# load model and tokenizer
access_token="hf_token"
model_path = "./dolly_pretrained_model"
model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)
#print(model)

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("./dolly_tokenizer", token=access_token)
tokenizer.pad_token = tokenizer.eos_token

# load default dataset that the model was trained on
dataset = load_from_disk("./dolly_dataset")
print(dataset)
print(dataset[0])
