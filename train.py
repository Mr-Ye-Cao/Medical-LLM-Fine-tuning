import torch
import argparse
from configs.lora_config import get_peft_config
from configs.training_args import get_sft_config
from src.data_utils import load_and_process_data
from src.modeling import load_model_and_tokenizer
from src.inference import interactive_test
from src.eval import DualEvaluationCallback, comprehensive_evaluation
from trl import SFTTrainer


def main(args):
    # Load model and tokenizer
    if args.resume_from_checkpoint:
        model, tokenizer = load_model_and_tokenizer(
            model_type=args.model_type,
            model_path=args.resume_from_checkpoint
        )
    else:
        model, tokenizer = load_model_and_tokenizer(model_type=args.model_type)

    # Load and process data
    dataset = load_and_process_data(model_type=args.model_type)

    if args.test:
        my_context = ' '.join(dataset["test"][args.input]['context']['contexts'])
        my_question = dataset["test"][args.input]['question']
        interactive_test(
            model,
            tokenizer,
            context=my_context,
            question=my_question,
            max_length=args.max_length,
            temperature=args.temperature
        )
        return

    # LoRA config (None for full fine-tuning)
    if args.full_finetune:
        peft_config = None
        print("Mode: Full fine-tuning (all parameters trainable)")
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params/1e6:.1f}M")
    else:
        peft_config = get_peft_config()
        print("Mode: LoRA fine-tuning")

    # SFT config (uses new TRL API)
    sft_config = get_sft_config(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        use_deepspeed=(args.model_type == "llama3-8b" and not args.full_finetune)
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )
    trainer.add_callback(DualEvaluationCallback(tokenizer, dataset["test"], args.model_type))

    if args.eval_only:
        trainer.evaluate()
        comprehensive_evaluation(trainer.model, tokenizer, dataset["test"], args.model_type)
    elif args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if not args.eval_only and not args.test:
        import os
        final_dir = os.path.join(args.output_dir, "final")

        if args.full_finetune:
            # Full fine-tuning: save the entire model to final/ subfolder
            trainer.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)
            print(f"Full model saved to {final_dir}")
        else:
            # LoRA: save adapter to final/ subfolder
            trainer.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)
            print(f"LoRA adapter saved to {final_dir}")

            # Merge LoRA weights into base model
            try:
                merged_model = trainer.model.merge_and_unload()
                merged_dir = os.path.join(args.output_dir, "merged")
                merged_model.save_pretrained(merged_dir, safe_serialization=True)
                tokenizer.save_pretrained(merged_dir)
                print(f"Merged model saved to {merged_dir}")
            except Exception as e:
                print(f"Could not merge model: {e}")

        # Optional: vLLM export (only for large models)
        if args.model_type == "llama3-8b" and args.export_vllm:
            try:
                from vllm import LLM
                vllm_model = LLM(
                    model=args.output_dir,
                    tokenizer=args.output_dir,
                    quantization="awq",
                    enforce_eager=True
                )
                vllm_model.save("./vllm_model")
                print("vLLM model saved to ./vllm_model")
            except ImportError:
                print("vLLM not installed, skipping vLLM export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model selection
    parser.add_argument("--model_type", type=str, default="olmo2-1b",
                        choices=["olmo2-1b", "llama3-8b"],
                        help="Model type to use")

    # Training method
    parser.add_argument("--full_finetune", action="store_true",
                        help="Full fine-tuning instead of LoRA (requires more GPU memory)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory")

    # Evaluation
    parser.add_argument("--eval_only", action="store_true",
                        help="Only run evaluation on test set")

    # Interactive test mode
    parser.add_argument("--test", action="store_true",
                        help="Interactive test mode")
    parser.add_argument("--input", type=int, default=0,
                        help="Test sample index")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")

    # Export options
    parser.add_argument("--export_vllm", action="store_true",
                        help="Export to vLLM format (llama3-8b only)")

    # DeepSpeed (for distributed training)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")

    args = parser.parse_args()

    # Set default output directory based on model type and training method
    if args.output_dir == "./results":
        method = "full" if args.full_finetune else "lora"
        args.output_dir = f"./results/{args.model_type}_pubmedqa_{method}"

    try:
        main(args)
    finally:
        # Distributed cleanup
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        # CUDA cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Cleanup complete")
