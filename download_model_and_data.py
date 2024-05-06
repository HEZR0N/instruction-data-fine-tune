# import libraries
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# load model and tokenizer
access_token="hf_token"
model_path = "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit"
model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)
model.save_pretrained("./openhermes_pretrained_model")
#print(model)

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.save_pretrained("./openhermes_tokenizer")
tokenizer.save_pretrained("./openhermes_pretrained_model")

# load default dataset that the model was trained on
# dataset = load_dataset("teknium/OpenHermes-2.5", split='train[:1%]', keep_in_memory=True)
dataset = load_dataset("jondurbin/airoboros-2.2", split='train[:10%]', keep_in_memory=True)
dataset.save_to_disk("./openhermes_dataset")
print(dataset)
print(dataset[0])

# load dataset for instruction_tuning
dataset_2 = load_dataset("ag_news")
dataset_2 = dataset_2['train'].select(range(3000))
dataset_2.save_to_disk("./news_dataset")
