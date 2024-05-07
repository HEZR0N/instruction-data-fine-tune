# Instruction Finetuning 
This repo was created for an assignment in my NLP/LLM class. The goal was to create a new instruction dataset (based on a pre-existing non-instruction dataset), fine-tuned model once with the new dataset and once with a combination of
the original dataset and the newset that the model was trained on, and examine the differences in performance.
The finetuned model was based off of:
 - https://huggingface.co/unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit
## Requirements
 - Python 3.11.8

Required Libraries/Modules made be installed with this command:
```
pip install transformers trl rouge bert_score evaluate nltk bitsandbytes xformers==0.0.25 peft==0.10.0 sentencepiece==0.2.0 protobuf==3.20.2 git+https://github.com/unslothai/unsloth
```

## Usage

1. Run `python download_model_and_data.py`
   - This will download the pretrained model and datasets
3. Run `python curate_instruction_dataset.py` to make the datasets ready to use for finetuning
4. Run `train_on_combined_datasets.py` to create a model finetuned on an instruction version of the `ag_news` dataset

The program is currently set to generate ouputs (one word text completions) for all data in the dataset, but only creates plots and evalutes metrics for the layers of the first example.

Below are the plots, metrics, and results that the program will output.

## Metrics
Uploading soon

## References
This medium blog was used as a reference to utilize the `unsloth` library:
`https://medium.com/@imranullahds/openchat-3-5-a-deep-dive-into-sft-fine-tuning-with-unsloth-0d9eba710571`
