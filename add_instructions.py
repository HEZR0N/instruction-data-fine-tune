# import libraries
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# load dataset for instruction_tuning
dataset_2 = load_from_disk("./news_dataset")

def add_label_name(data):
    data['label_name'] = {0:'World', 1:'Sports', 2:'Business', 3:'Science'}[data['label']]
    return data

def add_extra_instr_1(data):
    data['text'] = 'Answer with one word. Categorize the following passage with one of the following options: World, Sports, Business, or Science: ' + data['text']
    return data

def add_extra_instr_2(data):
    data['text'] = 'Answer with one word. Of the subjects, World, Sports, Business, and Science, Which subject would best describe the following passage: ' + data['text']
    return data


dataset_2 = dataset_2.map(add_label_name)
dataset_2 = dataset_2.map(add_extra_instr_1)
dataset_2.save_to_disk("./instruction_1_news_dataset")
dataset_3 = dataset_2.map(add_extra_instr_2)
dataset_3.save_to_disk("./instruction_2_news_dataset")
print(dataset_2[0])
print(dataset_3[0])
