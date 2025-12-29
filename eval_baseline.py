#!/usr/bin/env python3
"""
Evaluate baseline model (no fine-tuning) on PubMedQA test set.
"""

import torch
import argparse
import re
import time
from src.data_utils import load_and_process_data, SUBSETS
from src.modeling import load_model_and_tokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from rouge import Rouge


def evaluate_baseline(model, tokenizer, test_dataset, has_decision=True):
    """Run evaluation on test set."""
    model.eval()
    predictions = {"decisions": [], "answers": []}
    ground_truth = {"decisions": [], "answers": []}

    print(f"\nEvaluating on {len(test_dataset)} samples...")
    start_time = time.time()

    for i, example in enumerate(test_dataset):
        # Get the prompt (everything before ### Answer:)
        text = example["text"]
        prompt = text.split("### Answer:")[0] + "### Answer:\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract prediction
        if has_decision:
            match = re.search(r'Final Decision:\s*(\w+)', generated, re.I)
            pred_decision = match.group(1).lower() if match else "maybe"
            predictions["decisions"].append(pred_decision)
            ground_truth["decisions"].append(example.get("final_decision", "").lower())

        # Extract answer
        if "Long Answer:" in generated:
            pred_answer = generated.split("Long Answer:")[-1].strip()
        else:
            pred_answer = generated.split("### Answer:")[-1].strip() if "### Answer:" in generated else ""
        predictions["answers"].append(pred_answer)
        ground_truth["answers"].append(example.get("long_answer", ""))

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_dataset)}")

    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.1f}s ({len(test_dataset)/elapsed:.1f} samples/sec)")

    # Classification metrics
    if has_decision and predictions["decisions"]:
        print("\n=== Classification Metrics ===")
        acc = accuracy_score(ground_truth["decisions"], predictions["decisions"])
        f1 = f1_score(ground_truth["decisions"], predictions["decisions"], average="macro", zero_division=0)
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(ground_truth["decisions"], predictions["decisions"], zero_division=0))

    # Generation metrics
    print("\n=== Generation Metrics ===")
    valid_pairs = [
        (p, t) for p, t in zip(predictions["answers"], ground_truth["answers"])
        if p.strip() and t.strip()
    ]

    if valid_pairs:
        pred_valid, true_valid = zip(*valid_pairs)
        rouge = Rouge().get_scores(list(pred_valid), list(true_valid), avg=True)
        print(f"ROUGE-L F1: {rouge['rouge-l']['f']:.4f}")
        print(f"Valid predictions: {len(valid_pairs)}/{len(predictions['answers'])}")
    else:
        print("No valid predictions for ROUGE score")

    return {
        "accuracy": acc if has_decision else None,
        "f1": f1 if has_decision else None,
        "rouge_l": rouge['rouge-l']['f'] if valid_pairs else None
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on PubMedQA")
    parser.add_argument("--model_type", type=str, default="olmo2-1b",
                        choices=["olmo2-1b", "llama3-8b"])
    parser.add_argument("--subset", type=str, default="pqa_labeled",
                        choices=["pqa_labeled", "pqa_artificial", "pqa_unlabeled"])
    args = parser.parse_args()

    print(f"=== Baseline Evaluation ===")
    print(f"Model: {args.model_type}")
    print(f"Subset: {args.subset}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_type=args.model_type)

    # Load data (80/20 split)
    dataset = load_and_process_data(
        subset=args.subset,
        model_type=args.model_type
    )

    # Evaluate on test set
    has_decision = SUBSETS[args.subset]["has_decision"]
    results = evaluate_baseline(model, tokenizer, dataset["test"], has_decision)

    print("\n=== Summary ===")
    print(f"Model: {args.model_type} (baseline, no fine-tuning)")
    print(f"Dataset: {args.subset}")
    print(f"Test samples: {len(dataset['test'])}")
    if results["accuracy"] is not None:
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['f1']:.4f}")
    if results["rouge_l"] is not None:
        print(f"ROUGE-L: {results['rouge_l']:.4f}")


if __name__ == "__main__":
    main()
