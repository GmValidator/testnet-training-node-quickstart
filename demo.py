import os
from dataclasses import dataclass

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer

from dataset import SFTDataCollator, SFTDataset
from utils.constants import model2template

@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    warmup_steps: int
    gradient_checkpointing: bool
    weight_decay: float
    save_strategy: str
    eval_strategy: str  # Change evaluation_strategy to eval_strategy
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    early_stopping_patience: int

def train_lora(
    model_id: str, context_length: int, training_args: LoraTrainingArguments
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    # Apply LoRA adapter
    model = get_peft_model(model, lora_config)

    # Ensure all parameters require gradients
    for name, param in model.named_parameters():
        if param.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            param.requires_grad = True

    training_args_dict = {
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "warmup_steps": training_args.warmup_steps,
        "learning_rate": float(training_args.learning_rate),  # Ensure learning_rate is a float
        "bf16": True,
        "logging_steps": 20,
        "output_dir": "outputs",
        "optim": "paged_adamw_8bit",
        "remove_unused_columns": False,
        "num_train_epochs": training_args.num_train_epochs,
        "gradient_checkpointing": training_args.gradient_checkpointing,
        "weight_decay": training_args.weight_decay,
        "save_strategy": training_args.save_strategy,
        "eval_strategy": training_args.eval_strategy,  # Change evaluation_strategy to eval_strategy
        "load_best_model_at_end": training_args.load_best_model_at_end,
        "metric_for_best_model": training_args.metric_for_best_model,
        "greater_is_better": training_args.greater_is_better,
    }

    training_args = TrainingArguments(**training_args_dict)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )

    # Load dataset
    dataset = SFTDataset(
        file="data/demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    trainer.train()

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    # upload lora weights and tokenizer
    print("Training Completed.")

if __name__ == "__main__":
    # Define training arguments for LoRA fine-tuning
    training_args = LoraTrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=3e-5,
        warmup_steps=500,
        gradient_checkpointing=True,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",  # Change evaluation_strategy to eval_strategy
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        early_stopping_patience=2,
    )

    # Set model ID and context length
    model_id = "Qwen/Qwen1.5-0.5B"
    context_length = 2048

    # Start LoRA fine-tuning
    train_lora(
        model_id=model_id, context_length=context_length, training_args=training_args
    )
