# Instruction Finetuning 
This repo was created for an assignment in my NLP/LLM class. The goal was to create a new instruction dataset (based on a pre-existing non-instruction dataset), fine-tuned model once with the new dataset and once with a combination of
the original dataset and the newset that the model was trained on, and examine the differences in performance.
The finetuned model was based off of:
 - https://huggingface.co/unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit
## Requirements
 - Python 3.11.8

Required Libraries/Modules made be installed with this command:
```
pip install git+https://github.com/unslothai/unsloth triton xformers
```
