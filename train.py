import torch
import argparse
from configs.lora_config import get_peft_config
from configs.training_args import get_training_args
from src.data_utils import load_and_process_data, get_response_template
from src.modeling import load_model_and_tokenizer
from src.inference import interactive_test
from src.eval import DualEvaluationCallback, comprehensive_evaluation
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


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

    # LoRA config
    peft_config = get_peft_config()

    # Training args (conditionally use DeepSpeed for large models)
    training_args = get_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        use_deepspeed=(args.model_type == "llama3-8b")
    )

    # Data collator with model-specific response template
    response_template = get_response_template(args.model_type)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(response_template, add_special_tokens=False),
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        data_collator=collator,
        args=training_args,
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
        # Merge LoRA weights
        model = model.merge_and_unload()

        # Save HuggingFace format
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

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

    # Set default output directory based on model type
    if args.output_dir == "./results":
        args.output_dir = f"./results/{args.model_type}_pubmedqa"

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
