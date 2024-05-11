# Instruction Finetuning 
## Summary
This repo was created for an assignment in my NLP/LLM class. The objectives were to:
 - create a new instruction dataset of size 6000 (3000 `train` and 3000 `test`) (based on a pre-existing non-instruction dataset)
 - fine-tuned the pre-trained model once with the new instruction dataset
 - fine-tune the pre-trained model with a dataset that is a combination of the original dataset the pre-trained model was trained on and the new instruction dataset
 - examine the differences in performance across the pre-trained model, the model fine-tuned on the new instruction dataset, and the model fine-tuned on both the pre-trained model's dataset and the new instruction dataset

## Base Models and Datasets
 - [OpenHermes-2.5-Mistral-7B](https://huggingface.co/unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit): I chose the `OpenHermes-2.5` model specifically because the data it was trained on is publicly available and a speedy 4-bit version of the model is available.     
 - [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5): the dataset that `OpenHermes-2.5-Mistral-7B` was trained on
 - [airoboros-2.2](https://huggingface.co/datasets/jondurbin/airoboros-2.2) : a subset of the `OpenHermes-2.5-Mistral-7B` dataset, which is what will actually be used for training:
 - [ag_news](https://huggingface.co/datasets/ag_news): a non-instruction based dataset which will be turned into an instuction-based dataset for fine-tuning `OpenHermes-2.5-Mistral-7B` for text classifcation

## New Models and Datasets
 - Instruction-based dataset version of the `ag_news` dataset: [mixed_news_instruction_dataset](https://utsacloud-my.sharepoint.com/:f:/g/personal/hezron_perez_my_utsa_edu/ErkgebMbU9xBpjPhIdwkeRkBvRJcPabLh58lJcJ6I87HBg?e=QngS7m)
 - Model fine-tuned on the `mixed_news_instruction_dataset`: [openhermes_finetuned_model](https://utsacloud-my.sharepoint.com/:f:/g/personal/hezron_perez_my_utsa_edu/El7_4BlZFLJLmHGmyOQYPckBW3opf24mi3DGMxv5q2f4Dg?e=XkYmZp)
 - Model fine-tuned on the `mixed_news_instruction_dataset` and a subset of the `openhermes_dataset`: [openhermes_retune_finetuned_model](https://utsacloud-my.sharepoint.com/:f:/g/personal/hezron_perez_my_utsa_edu/EvN7Hmq0xsRHtK2PWLXcTfoBBGG9vWWcQwG8tiv6Y-C8jQ?e=W5Pphv)

## Requirements
 - Python 3.11.8

Required Libraries/Modules made be installed with this command:
```
pip install transformers trl rouge bert_score evaluate nltk bitsandbytes xformers==0.0.25 peft==0.10.0 sentencepiece==0.2.0 protobuf==3.20.2 git+https://github.com/unslothai/unsloth
```

## Usage

### Fine-tuning OpenHermes
You may go to the next section if you wish to use the fine-tuned models I provided in `New Models and Datasets` section above. Otherwise, to fine-tune the models yourself, run these steps: 

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
1. Run `python eval_with_mixed_news_instruction_dataset.py` to evaluate the 3 models. See `Evaluation Metrics` below

## Evaluation Metrics
The fine-tuned model performed the best, as expected. Surprisingly though, the re-tuned model performed the worst. I thin this might have to do with the fact that 2 training sets were so different (ie `mixed_news_instruction_dataset` had a one word response, while the OpenHermes-2.5 dataset had long paragraph responses.) such that objectives of the each dataset conflicted with one another, leading to a degredation in performance.

```
EVALUATING PRE-TRAINED MODEL
Expected: World			         Actual: Rating
Expected: Sports			 Actual: The
Expected: World			         Actual: Science
Expected: Science			 Actual: is
Expected: Business			 Actual: Science
Expected: Science			 Actual: Business
Expected: Business			 Actual: Business
Expected: Sports			 Actual: Sports
Expected: World			         Actual: World
Expected: Business			 Actual: Answer
		BLEU		ROUGE		BERTSCORE
		0.3000		0.3000		0.8167


EVALUATING FINE-TUNED MODEL
Expected: World		        	 Actual: Science
Expected: Sports			 Actual: Sports
Expected: World	        		 Actual: World
Expected: Science			 Actual: Science
Expected: Business			 Actual: Business
Expected: Science			 Actual: Science
Expected: Business			 Actual: Science
Expected: Sports			 Actual: Sports
Expected: World			         Actual: World
Expected: Business			 Actual: Business
		BLEU		ROUGE		BERTSCORE
		0.8000		0.8000		0.9549


EVALUATING RE-TUNED MODEL
Expected: World	        		 Actual: is
Expected: Sports			 Actual: is
Expected: World			         Actual: is
Expected: Science			 Actual: is
Expected: Business			 Actual: is
Expected: Science			 Actual: is
Expected: Business			 Actual: is
Expected: Sports			 Actual: is
Expected: World			         Actual: is
Expected: Business			 Actual: Business
		BLEU		ROUGE		BERTSCORE
		0.1000		0.1000		0.7273

```

## References
This medium blog was used as a reference to utilize the `unsloth` library:
`https://medium.com/@imranullahds/openchat-3-5-a-deep-dive-into-sft-fine-tuning-with-unsloth-0d9eba710571`
