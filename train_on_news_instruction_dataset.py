# import libraries
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer
import torch

# a token is not needed for the models and datasets implemented
access_token="hf_token"

# load model and tokenizer
model_path = "./openhermes_pretrained_model"
model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)
#print(model)

# initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("./openhermes_pretrained_model", token=access_token)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.save_pretrained("./openhermes_tokenizer")

# load new instruction dataset for fine-tuning
#dataset = load_from_disk("./openhermes_dataset")
dataset = load_from_disk("./mixed_news_instruction_dataset")
print(dataset)

device_map = "auto"
max_seq_length = 2048 
# Load OpenHermes with unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./openhermes_pretrained_model", # Supports OpenHermes
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)


# Add LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none", 
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False, 
    loftq_config = None, 
)

# Create the trainer
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
    dataset_text_field = "text", # dataset_text_field = "instruction",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 100,
        optim = "adamw_8bit",
        seed = 836,
        output_dir = "outputs",
        num_train_epochs=5
    ),
)


# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./openhermes_finetuned_model")
