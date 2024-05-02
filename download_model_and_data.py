# import libraries
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# load model and tokenizer
access_token="hf_token"
model_path = "databricks/dolly-v2-3b"
model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)
model.save_pretrained("./dolly_pretrained_model")
#print(model)

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, token=access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained("./dolly_tokenizer")

# load default dataset that the model was trained on
dataset = load_dataset("databricks/databricks-dolly-15k")
dataset = dataset['train'].select(range(15000))
dataset.save_to_disk("./dolly_dataset")
print(dataset)
print(dataset[0])

# load dataset for instruction_tuning
dataset_2 = load_dataset("ag_news")
dataset_2 = dataset_2['train'].select(range(3000))
dataset_2.save_to_disk("./news_dataset")
