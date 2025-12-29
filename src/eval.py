import re
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score, classification_report
from rouge import Rouge
from bert_score import score as bert_score

# Response templates for different models
RESPONSE_TEMPLATES = {
    "llama3-8b": "<|start_header_id|>assistant<|end_header_id|>\n",
    "olmo2-1b": "### Answer:\n",
}


class DualEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, test_dataset, model_type="olmo2-1b"):
        self.tokenizer = tokenizer
        self.test_dataset = test_dataset
        self.model_type = model_type

        response_template = RESPONSE_TEMPLATES.get(model_type, RESPONSE_TEMPLATES["olmo2-1b"])
        self.response_template = tokenizer.encode(
            response_template,
            add_special_tokens=False
        )

    def _extract_decision(self, text):
        """Extract classification label (yes/no/maybe)."""
        match = re.search(r'Final Decision:\s*(\w+)', text, re.IGNORECASE)
        return match.group(1).lower() if match else "maybe"

    def _extract_answer(self, text):
        """Extract long answer text."""
        parts = text.split("Long Answer:")
        return parts[1].strip() if len(parts) > 1 else ""

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.pop('model')
        pred_decisions, pred_answers = [], []
        true_decisions, true_answers = [], []

        for example in self.test_dataset:
            # Generate response
            inputs = self.tokenizer(example["text"], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=400)
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse results
            pred_decisions.append(self._extract_decision(full_text))
            pred_answers.append(self._extract_answer(full_text))
            true_decisions.append(example["final_decision"].lower())
            true_answers.append(example["long_answer"])

        # Classification metrics
        acc = accuracy_score(true_decisions, pred_decisions)
        f1 = f1_score(true_decisions, pred_decisions, average="macro")

        # Generation metrics (handle empty answers)
        valid_pairs = [(p, t) for p, t in zip(pred_answers, true_answers) if p.strip()]
        if valid_pairs:
            pred_valid, true_valid = zip(*valid_pairs)
            rouge = Rouge().get_scores(list(pred_valid), list(true_valid), avg=True)
            rouge_l = rouge['rouge-l']['f']
        else:
            rouge_l = 0.0

        print(f"Classification - Acc: {acc:.4f}, F1: {f1:.4f}")
        print(f"Generation - ROUGE-L: {rouge_l:.4f}")


def comprehensive_evaluation(model, tokenizer, test_dataset, model_type="olmo2-1b"):
    """Run comprehensive evaluation with detailed metrics."""
    model.eval()
    predictions = {"decisions": [], "answers": []}
    ground_truth = {"decisions": [], "answers": []}

    for example in test_dataset:
        inputs = tokenizer(example["text"], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse prediction
        match = re.search(r'Final Decision:\s*(\w+)', full_text, re.I)
        pred_decision = match.group(1).lower() if match else "maybe"
        pred_answer = full_text.split("Long Answer:")[1].strip() if "Long Answer:" in full_text else ""

        # Store results
        predictions["decisions"].append(pred_decision)
        predictions["answers"].append(pred_answer)
        ground_truth["decisions"].append(example["final_decision"].lower())
        ground_truth["answers"].append(example["long_answer"])

    # Classification report
    print("=== Classification Metrics ===")
    print(classification_report(ground_truth["decisions"], predictions["decisions"]))

    # Generation report
    print("\n=== Generation Metrics ===")

    # Filter empty predictions for ROUGE/BERTScore
    valid_pairs = [
        (p, t) for p, t in zip(predictions["answers"], ground_truth["answers"])
        if p.strip()
    ]

    if valid_pairs:
        pred_valid, true_valid = zip(*valid_pairs)
        pred_valid, true_valid = list(pred_valid), list(true_valid)

        rouge = Rouge().get_scores(pred_valid, true_valid, avg=True)
        bert_p, bert_r, bert_f = bert_score(pred_valid, true_valid, lang="en")

        print(f"ROUGE-L: {rouge['rouge-l']['f']:.4f}")
        print(f"BERTScore F1: {bert_f.mean().item():.4f}")
        print(f"Valid predictions: {len(valid_pairs)}/{len(predictions['answers'])}")
    else:
        print("No valid predictions for generation metrics")
