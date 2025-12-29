from transformers import TrainingArguments

################################################################################
# TrainingArguments parameters
################################################################################

# Batch size per GPU for training
per_device_train_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Optimizer to use
optim = "paged_adamw_32bit"

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Learning rate schedule
lr_scheduler_type = "cosine"

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 100

# Log every X updates steps
logging_steps = 20

# Eval settings
eval_strategy = "steps"
eval_steps = 50


def get_training_args(output_dir="./output", num_train_epochs=3, use_deepspeed=False):
    """
    Get training arguments.

    Args:
        output_dir: Directory for checkpoints and outputs
        num_train_epochs: Number of training epochs
        use_deepspeed: Enable DeepSpeed ZeRO-3 (for large models like LLaMA-3-8B)

    Returns:
        TrainingArguments
    """
    args_dict = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=max_grad_norm,
        optim=optim,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        group_by_length=group_by_length,
        bf16=True,  # Use bfloat16 for modern GPUs
        fp16=False,
    )

    # Add DeepSpeed config for large models
    if use_deepspeed:
        args_dict["deepspeed"] = "./configs/deepspeed_z3.json"

    return TrainingArguments(**args_dict)
