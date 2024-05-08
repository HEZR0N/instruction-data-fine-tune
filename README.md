# Instruction Finetuning 
This repo was created for an assignment in my NLP/LLM class. The objectives were to:
 - create a new instruction dataset (based on a pre-existing non-instruction dataset)
 - fine-tuned the pre-trained model once with the new instruction dataset
 - fine-tune the pre-trained model with a dataset that is a combination of hte original dataset the pre-trained model was trained on and the new instruction dataset
 - examine the differences in performance across the pre-trained model, the model fine-tuned on the new instruction dataset, and the model fine-tuned on both the pre-trained model's dataset and the new instruction dataset

The original pre-trained model which will be fine-tuned is:
 - https://huggingface.co/unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit    
`OpenHermes-2.5` was trained off of this dataset
 - https://huggingface.co/datasets/teknium/OpenHermes-2.5
However, for training, we will only use a subset of that dataset:
 - https://huggingface.co/datasets/jondurbin/airoboros-2.2
The dataset that will be used to fine-tune `OpenHermes-2.5` will be based on this dataset
 - https://huggingface.co/datasets/ag_news

The links to the fine-tuned versions of the `OpenHermes-2.5` model can be found in `MODEL_LINKS.md`
## Requirements
 - Python 3.11.8

Required Libraries/Modules made be installed with this command:
```
pip install transformers trl rouge bert_score evaluate nltk bitsandbytes xformers==0.0.25 peft==0.10.0 sentencepiece==0.2.0 protobuf==3.20.2 git+https://github.com/unslothai/unsloth
```

## Usage

### Fine-tuning OpenHermes
You may go to the next section if you wish to use the fine-tuned models I provided in `MODEL_LINKS.md`. Otherwise, to fine-tune the models yourself, run these steps. 

1. Run `python download_model_and_data.py` to download and save the model and datasets locally
 - This will download the pretrained `OpenHermes-2.5-Mistral-7B` model and 2 datasets
   - The `OpenHermes-2.5-Mistral-7B` model was trained on a dataset (`teknium/OpenHermes-2.5`) with 1,000,000 elements, which itself is comprised of other smaller datasets. Due to time constraints, we will one of the sub-datasets, `airoboros-2.2`
   - The `ag_news` dataset is collection of sentences from news articles
2. Run `python curate_instruction_dataset.py` to modify `ag_news` dataset to become an instruction-baed dataset for finetuning saved as folder `mixed_news_instruction_dataset`
4. Run `train_on_news_instruction_dataset.py` to train the `OpenHermes` model on the instruction-based `ag_news` dataset saved as folder `openhermes_finetuned_model`
   - This will create a finetuned version of the `OpenHermes` model
4. Run `train_on_combined_datasets.py` to train the `OpenHermes` model on the a dataset made of instruction-based `ag_news` dataset and the `airoboros-2.2` dataset saved as folder `openhermes_retune_finetuned_model`
   - This will create a different finetuned version of the `OpenHermes` model
 ### Evaluating the Models
 Uploading soon

## Metrics
Uploading soon

## References
This medium blog was used as a reference to utilize the `unsloth` library:
`https://medium.com/@imranullahds/openchat-3-5-a-deep-dive-into-sft-fine-tuning-with-unsloth-0d9eba710571`
