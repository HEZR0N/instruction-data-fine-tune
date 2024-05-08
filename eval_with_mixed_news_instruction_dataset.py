# import libraries
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt')
from rouge import Rouge
from bert_score import BERTScorer
from evaluate import load
import matplotlib.pyplot as plt
from statistics import mean
roug = Rouge()
bertscore = load("bertscore")

# load new instruction dataset for testing each model's performance
dataset = load_from_disk("./mixed_news_instruction_dataset")
print(dataset)

device_map = "auto"
max_seq_length = 2048 

def load_model(model_path):
  # Load a version of OpenHermes with unsloth
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name = model_path, # Supports OpenHermes
      max_seq_length = max_seq_length,
      dtype = None,
      load_in_4bit = True,
  )
  return model, tokenizer

def get_model_outputs(model, tokenizer):
  model_outputs = []
  counter = 0
  for i in dataset['test']['instruction'][:10]:
    with torch.no_grad():
      tokens = tokenizer(i, return_tensors='pt', padding=True, max_length=1024)
      output = model.generate(**tokens, num_return_sequences=1, return_dict_in_generate=True, output_logits=True, output_hidden_states=True, max_new_tokens=7, pad_token_id=tokenizer.eos_token_id)
      decoded_ouput = tokenizer.batch_decode(output[0], skip_special_tokens=True)
      decoded_ouput = decoded_ouput[0].replace('[/b]', '').replace('/', '').replace('[', '').replace(':', '').replace('.', '').replace(',', '').split()[-1]
      model_outputs.append(decoded_ouput)
      print(f"Expected: {dataset['test']['label_name'][counter]}\t\t\t Actual: {decoded_ouput}")
      counter += 1
  return model_outputs

def get_metrics(new_response, ground_truth):
  blue_weights = (1.0, 0.0, 0.0, 0.0, 0.0)
  blue = nltk.translate.bleu_score.sentence_bleu([ground_truth.split()], new_response.split(), blue_weights)
  red = roug.get_scores(new_response, ground_truth)
  red = red[0]['rouge-l']['f']
  raw_bert = bertscore.compute(predictions=[new_response], references=[ground_truth], model_type="distilbert-base-uncased")
  bert = raw_bert['f1'][0]
  return blue, red, bert

def avg_metrics(outputs, dataset):
  bleus = []
  reds = []
  berts = []
  for i in range(len(outputs)):
    blue, red, bert = get_metrics(outputs[i], dataset['test']['label_name'][i])
    bleus.append(blue)
    reds.append(red)
    berts.append(bert)
  print("\t\tBLEU\t\tROUGE\t\tBERTSCORE")
  print(f"\t\t{mean(bleus):.4f}\t\t{mean(reds):.4f}\t\t{mean(berts):.4f}")

print("EVALUATING PRE-TRAINED MODEL")
model, tokenizer = load_model('./openhermes_pretrained_model')
pre_trained_outputs = get_model_outputs(model, tokenizer)
avg_metrics(pre_trained_outputs, dataset)

print("EVALUATING FINE-TUNED MODEL")
model, tokenizer = load_model('./openhermes_finetuned_model')
fine_tuned_outputs = get_model_outputs(model, tokenizer)
avg_metrics(fine_tuned_outputs, dataset)

print("EVALUATING RE-TUNED MODEL")
model, tokenizer = load_model('./openhermes_retune_finetuned_model')
re_tuned_outputs = get_model_outputs(model, tokenizer)
avg_metrics(re_tuned_outputs, dataset)
