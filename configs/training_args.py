from trl import SFTConfig

################################################################################
# SFTConfig parameters (new TRL API)
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

# Max sequence length
max_length = 1024

# Dataset text field
dataset_text_field = "text"


def get_sft_config(output_dir="./output", num_train_epochs=3, use_deepspeed=False,
                   save_all_checkpoints=False, save_every_epoch=False):
    """
    Get SFT config (new TRL API).

    Args:
        output_dir: Directory for checkpoints and outputs
        num_train_epochs: Number of training epochs
        use_deepspeed: Enable DeepSpeed ZeRO-3 (for large models like LLaMA-3-8B)
        save_all_checkpoints: If True, save all checkpoints (no limit)
        save_every_epoch: If True, save checkpoint at each epoch end

    Returns:
        SFTConfig
    """
    config_dict = dict(
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
        save_total_limit=None if save_all_checkpoints else 3,
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable tensorboard for now
        group_by_length=group_by_length,
        bf16=True,  # Use bfloat16 for modern GPUs
        fp16=False,
        # SFT-specific settings
        max_length=max_length,
        dataset_text_field=dataset_text_field,
        packing=False,  # Disable packing for simplicity
    )

    # Save at each epoch end
    if save_every_epoch:
        config_dict["save_strategy"] = "epoch"
        config_dict["eval_strategy"] = "epoch"
        del config_dict["save_steps"]
        del config_dict["eval_steps"]

    # Add DeepSpeed config for large models
    if use_deepspeed:
        config_dict["deepspeed"] = "./configs/deepspeed_z3.json"

    return SFTConfig(**config_dict)


# Keep old function for backwards compatibility
def get_training_args(output_dir="./output", num_train_epochs=3, use_deepspeed=False):
    """Deprecated: Use get_sft_config instead."""
    return get_sft_config(output_dir, num_train_epochs, use_deepspeed)
