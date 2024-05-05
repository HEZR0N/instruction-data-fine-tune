# import libraries
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# load dataset for instruction_tuning
dataset = load_from_disk("./news_dataset")

# add the classification labels to the dataset, which are currently just represented by numbers
def add_label_name(data):
    data['label_name'] = {0:'World', 1:'Sports', 2:'Business', 3:'Science'}[data['label']]
    return data

# append the first type of instruction to data
def add_extra_instr_1(data):
    data['text'] = 'Answer with one word. Categorize the following passage with one of the following options: World, Sports, Business, or Science: ' + data['text']
    return data

# append the second type of instruction to data
def add_extra_instr_2(data):
    data['text'] = 'Answer with one word. Of the subjects, World, Sports, Business, and Science, which subject would best describe the following passage: ' + data['text']
    return data

# call label function
dataset = dataset.map(add_label_name)

# Ideally, there would just be two columns, one for each instruction for each elemtent, but instead I made 2 datasets for because each element from the dataset should
# work with 2 different types of instructions

# Save the first version of the dataset where each element has the first type of instruction
dataset_1 = dataset.map(add_extra_instr_1)
dataset_1.save_to_disk("./instruction_1_news_dataset")

# Save the second version of the dataset where each element has the second type of instruction
dataset_2 = dataset.map(add_extra_instr_2)
dataset_2.save_to_disk("./instruction_2_news_dataset")

# show the two types of instructions for the first element of the dataset
print(dataset_1[0])
print(dataset_2[0])
