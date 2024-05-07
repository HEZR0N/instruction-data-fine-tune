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

### Fine-tuning OpenHermes
1. Run `python download_model_and_data.py` to download and save the model and datasets locally
 - This will download the pretrained `OpenHermes-2.5-Mistral-7B` model and 2 datasets
   - The `OpenHermes-2.5-Mistral-7B` model was trained on a dataset (`teknium/OpenHermes-2.5`) with 1,000,000 elements, which itself is comprised of other smaller datasets. Due to time constraints, we will one of the sub-datasets, `airoboros-2.2`
   - The `ag_news` dataset is collection of sentences from news articles
2. Run `python curate_instruction_dataset.py` to modify `ag_news` dataset to become an instruction-baed dataset for finetuning saved as folder `mixed_news_instruction_dataset`
4. Run `train_on_news_instruction_dataset.py` to train the `OpenHermes` model on the instruction-based `ag_news` dataset saved as folder `openhermes_finetuned_model`
   - This will create a finetuned version of the `OpenHermes` model
4. Run `train_on_combined_datasets.py` to train the `OpenHermes` model on the instruction-based `ag_news` dataset and the dataset openhermes was initially trained on saved as folder `openhermes_retune_finetuned_model`
   - This will create a different finetuned version of the `OpenHermes` model
 ### Evaluating the Models

The program is currently set to generate ouputs (one word text completions) for all data in the dataset, but only creates plots and evalutes metrics for the layers of the first example.

Below are the plots, metrics, and results that the program will output.

## Metrics
Uploading soon

## References
This medium blog was used as a reference to utilize the `unsloth` library:
`https://medium.com/@imranullahds/openchat-3-5-a-deep-dive-into-sft-fine-tuning-with-unsloth-0d9eba710571`
