# import libraries
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# a token is not needed for the models and datasets implemented
access_token = "hf_koken"

# load model and tokenizer
model_path = "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit"
model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)
model.save_pretrained("./openhermes_pretrained_model")
#print(model)

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained("./openhermes_pretrained_model")

# load default dataset that the model was trained on
# not using 1,000,000 OpenHermes-2.5 data, but instead airoboros data, which is a part of OpenHermes-2.5
# dataset = load_dataset("teknium/OpenHermes-2.5", split='train[:1%]', keep_in_memory=True)
dataset = load_dataset("jondurbin/airoboros-2.2", split='train[:3000]')
dataset.save_to_disk("./openhermes_dataset")
print(dataset)
#print(dataset[0])

# load dataset for instruction_tuning
new_dataset = load_dataset("ag_news")
new_dataset = dataset_2['train'].select(range(3000))
new_dataset.save_to_disk("./news_dataset")
print(new_dataset)
print(new_dataset[0])
