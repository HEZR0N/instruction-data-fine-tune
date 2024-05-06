# import libraries
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

# load model and tokenizer
access_token="hf_token"
model_path = "./openhermes_pretrained_model"
model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)
#model.save_pretrained("./openhermes_pretrained_model")
#print(model)

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("./openhermes_tokenizer", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.save_pretrained("./openhermes_tokenizer")
#exit(0)

# load default dataset that the model was trained on
dataset = load_from_disk("./openhermes_dataset")
print(dataset)
#print(dataset[0])


device_map = "auto"
max_seq_length = 2048
#
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./openhermes_pretrained_model", # Supports OpenHermes
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
