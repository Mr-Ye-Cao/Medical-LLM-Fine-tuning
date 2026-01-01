#!/usr/bin/env python3
"""
Fast baseline evaluation using vLLM for batch inference.
~10-20x faster than sequential transformers generation.
"""

import argparse
import re
import time
from vllm import LLM, SamplingParams
from src.data_utils import load_and_process_data, SUBSETS, get_response_template
from sklearn.metrics import accuracy_score, f1_score, classification_report
from rouge import Rouge


def evaluate_with_vllm(model_path, test_dataset, model_type, has_decision=True):
    """Run evaluation using vLLM for fast batch inference."""

    response_template = get_response_template(model_type)
    print(f"Response template: {repr(response_template)}")

    # Prepare prompts
    prompts = []
    ground_truth = {"decisions": [], "answers": []}

    for example in test_dataset:
        text = example["text"]
        prompt = text.split(response_template)[0] + response_template
        prompts.append(prompt)

        if has_decision:
            ground_truth["decisions"].append(example.get("final_decision", "").lower())
        ground_truth["answers"].append(example.get("long_answer", ""))

    print(f"\nPrepared {len(prompts)} prompts for batch inference...")

    # Initialize vLLM
    print(f"Loading model with vLLM: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy decoding
        max_tokens=200,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Batch generation
    print("Running batch inference...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time

    print(f"Batch inference completed in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} samples/sec)")

    # Process outputs
    predictions = {"decisions": [], "answers": []}

    for output in outputs:
        generated = output.outputs[0].text.strip()

        # Extract decision
        if has_decision:
            match = re.search(r'Final Decision:\s*(\w+)', generated, re.I)
            pred_decision = match.group(1).lower() if match else "maybe"
            predictions["decisions"].append(pred_decision)

        # Extract answer
        if "Long Answer:" in generated:
            pred_answer = generated.split("Long Answer:")[-1].strip()
        else:
            pred_answer = generated
        predictions["answers"].append(pred_answer)

    # Classification metrics
    results = {}
    if has_decision and predictions["decisions"]:
        print("\n=== Classification Metrics ===")
        acc = accuracy_score(ground_truth["decisions"], predictions["decisions"])
        f1 = f1_score(ground_truth["decisions"], predictions["decisions"], average="macro", zero_division=0)
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(ground_truth["decisions"], predictions["decisions"], zero_division=0))
        results["accuracy"] = acc
        results["f1"] = f1

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
        results["rouge_l"] = rouge['rouge-l']['f']
    else:
        print("No valid predictions for ROUGE score")
        results["rouge_l"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Fast baseline evaluation using vLLM")
    parser.add_argument("--model_type", type=str, default="olmo3-7b-instruct",
                        choices=["olmo2-1b", "llama3-8b", "olmo3-7b-instruct"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="Custom model path (e.g., fine-tuned checkpoint)")
    parser.add_argument("--subset", type=str, default="pqa_labeled",
                        choices=["pqa_labeled", "pqa_artificial", "pqa_unlabeled"])
    args = parser.parse_args()

    # Default model paths
    MODEL_PATHS = {
        "olmo2-1b": "../OLMo-2-0425-1B",
        "olmo3-7b-instruct": "../OLMo-3-7B-Instruct",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    }

    model_path = args.model_path or MODEL_PATHS.get(args.model_type)

    print(f"=== vLLM Baseline Evaluation ===")
    print(f"Model: {args.model_type}")
    print(f"Model path: {model_path}")
    print(f"Subset: {args.subset}")

    # Load data
    dataset = load_and_process_data(
        subset=args.subset,
        model_type=args.model_type
    )

    # Evaluate
    has_decision = SUBSETS[args.subset]["has_decision"]
    results = evaluate_with_vllm(model_path, dataset["test"], args.model_type, has_decision)

    print("\n=== Summary ===")
    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.subset}")
    print(f"Test samples: {len(dataset['test'])}")
    if results.get("accuracy") is not None:
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['f1']:.4f}")
    if results.get("rouge_l") is not None:
        print(f"ROUGE-L: {results['rouge_l']:.4f}")


if __name__ == "__main__":
    main()
