#!/usr/bin/env python3
"""
Official PubMedQA evaluation using vLLM for batch inference.
Uses the official 500 sample test set and outputs predictions in official format.
"""

import argparse
import json
import os
import re
import time
from vllm import LLM, SamplingParams
from src.data_utils import load_and_process_data, SUBSETS, get_response_template, OFFICIAL_TEST_PATH
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_official(model_path, test_dataset, model_type):
    """Run evaluation using vLLM on official PubMedQA test set."""

    response_template = get_response_template(model_type)
    print(f"Response template: {repr(response_template)}")

    # Load official ground truth
    with open(OFFICIAL_TEST_PATH) as f:
        official_gt = json.load(f)
    print(f"Official test set: {len(official_gt)} samples")

    # Prepare prompts with PMIDs
    prompts = []
    pmids = []
    ground_truth = []

    for example in test_dataset:
        pmid = str(example['pubid'])
        if pmid not in official_gt:
            continue

        text = example["text"]
        prompt = text.split(response_template)[0] + response_template
        prompts.append(prompt)
        pmids.append(pmid)
        ground_truth.append(official_gt[pmid])

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
    predictions = {}
    pred_list = []

    for i, output in enumerate(outputs):
        generated = output.outputs[0].text.strip()

        # Extract decision (yes/no/maybe)
        match = re.search(r'Final Decision:\s*(yes|no|maybe)', generated, re.I)
        if match:
            pred = match.group(1).lower()
        else:
            # Fallback: check for keywords
            gen_lower = generated.lower()
            if 'yes' in gen_lower[:50]:
                pred = 'yes'
            elif 'no' in gen_lower[:50]:
                pred = 'no'
            else:
                pred = 'maybe'

        predictions[pmids[i]] = pred
        pred_list.append(pred)

    # Calculate metrics
    print("\n=== Official PubMedQA Metrics ===")
    acc = accuracy_score(ground_truth, pred_list)
    maf = f1_score(ground_truth, pred_list, average='macro')

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {maf:.4f}")
    print("\nClassification Report:")
    print(classification_report(ground_truth, pred_list, labels=['yes', 'no', 'maybe']))

    # Distribution analysis
    print("\n=== Prediction Distribution ===")
    for label in ['yes', 'no', 'maybe']:
        gt_count = ground_truth.count(label)
        pred_count = pred_list.count(label)
        print(f"{label}: GT={gt_count}, Pred={pred_count}")

    return {
        "accuracy": acc,
        "macro_f1": maf,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Official PubMedQA evaluation using vLLM")
    parser.add_argument("--model_type", type=str, default="olmo3-7b-instruct",
                        choices=["olmo2-1b", "llama3-8b", "olmo3-7b-instruct"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="Custom model path (e.g., fine-tuned checkpoint)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save predictions to JSON file (for official submission)")
    args = parser.parse_args()

    # Default model paths
    MODEL_PATHS = {
        "olmo2-1b": "../OLMo-2-0425-1B",
        "olmo3-7b-instruct": "../OLMo-3-7B-Instruct",
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    }

    model_path = args.model_path or MODEL_PATHS.get(args.model_type)

    print(f"=== Official PubMedQA Evaluation ===")
    print(f"Model: {args.model_type}")
    print(f"Model path: {model_path}")

    # Load data with official split
    dataset = load_and_process_data(
        subset="pqa_labeled",
        model_type=args.model_type,
        use_official_split=True
    )

    # Evaluate
    results = evaluate_official(model_path, dataset["test"], args.model_type)

    print("\n=== Summary ===")
    print(f"Model: {args.model_type}")
    print(f"Test samples: 500 (official)")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro-F1: {results['macro_f1']:.4f}")

    # Save predictions if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results['predictions'], f, indent=2)
        print(f"\nPredictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
