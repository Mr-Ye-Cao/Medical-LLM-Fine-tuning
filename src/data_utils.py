import json
import os
from datasets import load_dataset, Dataset

# Available PubMedQA subsets
SUBSETS = {
    "pqa_labeled": {"size": 1000, "has_decision": True},
    "pqa_artificial": {"size": 211269, "has_decision": True},
    "pqa_unlabeled": {"size": 61249, "has_decision": False},
}

# Path to official PubMedQA test ground truth
OFFICIAL_TEST_PATH = os.path.join(os.path.dirname(__file__), "../../pubmedqa/data/test_ground_truth.json")

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
    "olmo3-7b-instruct": {
        "format": (
            "<|im_start|>system\n{system_msg}\n{context_msg}<|im_end|>\n"
            "<|im_start|>user\n{user_input}<|im_end|>\n"
            "<|im_start|>assistant\n{assistant_response}<|im_end|>"
        ),
        "response_template": "<|im_start|>assistant\n",
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


def load_and_process_data(
    dataset_name="qiaojin/PubMedQA",
    subset="pqa_labeled",
    test_size=0.2,
    model_type="olmo2-1b",
    use_official_split=False
):
    """
    Load and process PubMedQA dataset.

    Args:
        dataset_name: HuggingFace dataset name
        subset: One of "pqa_labeled", "pqa_artificial", "pqa_unlabeled"
        test_size: Fraction for test split (default 0.2 = 80/20 split)
        model_type: One of "llama3-8b" or "olmo2-1b"
        use_official_split: If True, use official PubMedQA 500/500 train/test split

    Returns:
        Processed dataset with train/test splits
    """
    if model_type not in TEMPLATES:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {list(TEMPLATES.keys())}")

    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset: {subset}. Choose from {list(SUBSETS.keys())}")

    template = TEMPLATES[model_type]["format"]
    has_decision = SUBSETS[subset]["has_decision"]

    print(f"Loading {subset} from {dataset_name}...")
    full_dataset = load_dataset(dataset_name, subset)["train"]

    if use_official_split and subset == "pqa_labeled":
        # Use official PubMedQA test split (500 train / 500 test)
        if not os.path.exists(OFFICIAL_TEST_PATH):
            raise FileNotFoundError(f"Official test ground truth not found: {OFFICIAL_TEST_PATH}")

        with open(OFFICIAL_TEST_PATH) as f:
            test_ground_truth = json.load(f)
        test_pmids = set(test_ground_truth.keys())

        # Split by PMID
        train_examples = []
        test_examples = []
        for ex in full_dataset:
            if str(ex['pubid']) in test_pmids:
                test_examples.append(ex)
            else:
                train_examples.append(ex)

        from datasets import DatasetDict
        dataset = DatasetDict({
            "train": Dataset.from_list(train_examples),
            "test": Dataset.from_list(test_examples)
        })
        print(f"Using official PubMedQA split: {len(train_examples)} train, {len(test_examples)} test")
    else:
        dataset = full_dataset.train_test_split(test_size=test_size, seed=42)

    def format_instruction(example):
        context = ' '.join(example['context']['contexts'])

        # Handle unlabeled data (no final_decision)
        if has_decision:
            answer = f"Final Decision: {example['final_decision']}\nLong Answer: {example['long_answer']}"
        else:
            answer = f"Long Answer: {example['long_answer']}"

        return {
            "text": template.format(
                system_msg=SYSTEM_MSG,
                context_msg=f"Context: {context}",
                user_input=example['question'],
                assistant_response=answer
            )
        }

    formatted = dataset.map(format_instruction)
    print(f"Dataset loaded: {len(formatted['train'])} train, {len(formatted['test'])} test")
    print(f"Subset: {subset}, Has decision labels: {has_decision}")

    return formatted
