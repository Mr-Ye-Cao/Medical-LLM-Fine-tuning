from datasets import load_dataset

# Instruction templates for different models
TEMPLATES = {
    "llama3-8b": {
        "format": (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n{system_msg}\n{context_msg}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n{assistant_response}<|eot_id|>"
        ),
        "response_template": "<|start_header_id|>assistant<|end_header_id|>\n",
    },
    "olmo2-1b": {
        "format": (
            "### System:\n{system_msg}\n{context_msg}\n"
            "### Question:\n{user_input}\n"
            "### Answer:\n{assistant_response}"
        ),
        "response_template": "### Answer:\n",
    },
}

SYSTEM_MSG = (
    "You are a clinical expert. Your task is to analyze the given medical literature context "
    "and then provide a Final Decision and a Long Answer."
)


def get_response_template(model_type="olmo2-1b"):
    """Get the response template for DataCollatorForCompletionOnlyLM."""
    if model_type not in TEMPLATES:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(TEMPLATES.keys())}")
    return TEMPLATES[model_type]["response_template"]


def load_and_process_data(dataset_name="qiaojin/PubMedQA", test_size=0.1, model_type="olmo2-1b"):
    """
    Load and process PubMedQA dataset.

    Args:
        dataset_name: HuggingFace dataset name
        test_size: Fraction for test split
        model_type: One of "llama3-8b" or "olmo2-1b"

    Returns:
        Processed dataset with train/test splits
    """
    if model_type not in TEMPLATES:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(TEMPLATES.keys())}")

    template = TEMPLATES[model_type]["format"]

    dataset = load_dataset(dataset_name, "pqa_labeled")
    dataset = dataset["train"].train_test_split(test_size=test_size, seed=42)

    def format_instruction(example):
        context = ' '.join(example['context']['contexts'])
        return {
            "text": template.format(
                system_msg=SYSTEM_MSG,
                context_msg=f"Context: {context}",
                user_input=example['question'],
                assistant_response=f"Final Decision: {example['final_decision']}\nLong Answer: {example['long_answer']}"
            )
        }

    formatted = dataset.map(format_instruction)
    print(f"Dataset loaded: {len(formatted['train'])} train, {len(formatted['test'])} test")

    return formatted
