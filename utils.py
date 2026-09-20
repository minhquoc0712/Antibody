import os
import io
import json
import torch.nn as nn
import torch
import numpy as np
import copy
from torch.utils.data import Dataset
import transformers
from typing import Dict, Optional, Sequence
import logging
from loguru import logger
import random

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict



def jload_jsonl(f, mode="r"):
    """Load a .jsonl file.

    The function is flexible:
    • If a line contains valid JSON, we parse it with ``json.loads`` and append the resulting object.
    • If parsing fails, we treat the whole line as a raw string.  This allows
      using simple newline-separated text files as a source of refusal answers.
    Returns a list of objects (dicts / strings).
    """
    f = _make_r_io_base(f, mode)
    objs = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError:
            objs.append(line)
    f.close()
    return objs


def track_embedding(extra_args, eval_dataloader, model, track_batch_number=100):
    from transformers.models.llama.modeling_llama import LlamaAttention

    model.eval()
    # save alignment embedding
    alignment_embedding = [{} for i in range(track_batch_number)]
    for index, batch in enumerate(eval_dataloader):
        if index < track_batch_number:
            hooks = []
            alignment_embedding_per_data = alignment_embedding[index]

            # Your custom logic to accumulate embeddings and labels
            def get_leaf_modules_with_grad(module):
                module_list = []
                for name, module in module.named_modules():
                    if isinstance(module, LlamaAttention):
                        module.name = name
                        module_list += [module]
                # # print(module_list)
                return module_list

            def track_embedding_hook(module, input, output):
                # if torch.norm(output[0].detach().to("cpu")) <100000:
                alignment_embedding_per_data[module.name] = output[0].detach().to("cpu")
                # print(output.shape)
                # print(module.name)
                # print(torch.norm(alignment_embedding_per_data[module.name]))
                print(output[0].isnan().any())
                torch.cuda.empty_cache()
                return output

            leaf_modules_with_grad = get_leaf_modules_with_grad(model)
            for layer in leaf_modules_with_grad:
                hook = layer.register_forward_hook(track_embedding_hook)
                hooks.append(hook)

            inputs = batch["input_ids"]
            outputs = model(inputs)
            for hook in hooks:
                hook.remove()
            hooks = []
    torch.save(alignment_embedding, extra_args.lora_folder + "/alignment_embedding.pt")


def calculate_drift2first_embedding(
    extra_args, eval_dataloader, model, track_batch_number=100
):
    from transformers.models.llama.modeling_llama import LlamaAttention

    model.eval()
    # first read initial represnetation
    alignment_embedding = torch.load(extra_args.lora_folder + "/alignment_embedding.pt")
    drift = 0
    for index, batch in enumerate(eval_dataloader):
        if index < track_batch_number:
            hooks = []
            alignment_embedding_per_data = alignment_embedding[index]

            # Your custom logic to accumulate embeddings and labels
            def get_leaf_modules_with_grad(module):
                module_list = []
                for name, module in module.named_modules():
                    if isinstance(module, LlamaAttention):
                        module.name = name
                        module_list += [module]
                return module_list

            def track_drift_hook(module, input, output):
                nonlocal drift
                if module.name in alignment_embedding_per_data:
                    output_float32 = output[0].detach().to("cpu").to(torch.float32)
                    alignment_float32 = alignment_embedding_per_data[module.name].to(
                        torch.float32
                    )
                    drift += torch.norm(output_float32 - alignment_float32)
                torch.cuda.empty_cache()
                return output

            leaf_modules_with_grad = get_leaf_modules_with_grad(model)
            for layer in leaf_modules_with_grad:
                hook = layer.register_forward_hook(track_drift_hook)
                hooks.append(hook)

            inputs = batch["input_ids"]
            outputs = model(inputs)
            for hook in hooks:
                hook.remove()
            hooks = []
    print("embedding drift {}".format(drift))


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    IGNORE_INDEX = -100
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def extract_beavertails_safe_response(example):
    split_text = example["refusal"].split("\nAnswer: ")
    question = split_text[0].replace("Question: ", "")
    answer = split_text[1]
    return {"instruction": question, "input": "", "output": answer}


def extract_beavertails_harmful_response(example):
    return {
        "instruction": example["prompt"],
        "input": "",
        "output": example["response"],
    }


class SupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        poison_ratio=None,
        sample_num=None,
        benign_dataset=None,
        finetuning_guide_data_num=None,
        data_start_index=5000,
        seed=42,
        refusal_data_path: str = None,
        add_harmful_flag: bool = False,
    ):
        super().__init__()
        self.add_harmful_flag = add_harmful_flag

        logger.info(f"Loading data from {data_path}...")
        logger.info(f"Poison ratio: {poison_ratio}")
        logger.info(f"Sample num: {sample_num}")
        logger.info(f"Benign dataset: {benign_dataset}")
        logger.info(f"Finetuning guide data num: {finetuning_guide_data_num}")
        logger.info(f"Data start index: {data_start_index}")
        logger.info(f"Refusal data path: {refusal_data_path}")

        # Load alignment safe data.
        if "beavertails_with_refusals_train_filtered_safe" in data_path:
            list_data_dict = []
            dataset = load_subset("data/beavertails_with_refusals_train_filtered.json", data_start_index, sample_num)
            for example in dataset:
                instance = extract_beavertails_safe_response(example)
                if self.add_harmful_flag:
                    # Mark sample as benign (safe)
                    instance["is_harmful"] = 1
                list_data_dict += [instance]
        # Load alignment harmful data
        elif "beavertails_with_refusals_train_filtered_harmful" in data_path:
            list_data_dict = []
            dataset = load_subset("data/beavertails_with_refusals_train_filtered.json", data_start_index, sample_num)
            for example in dataset:
                instance = extract_beavertails_harmful_response(example)
                if self.add_harmful_flag:
                    # Mark sample as harmful
                    instance["is_harmful"] = 1
                list_data_dict += [instance]
        # Load harmful finetuning data (poisoned and benign)
        elif "beavertails_disjoint_attack_deduplicated" in data_path:
            list_data_dict = []
            poison_num = int(poison_ratio * sample_num)
            dataset = load_subset("data/beavertails_disjoint_attack_deduplicated.json", data_start_index, poison_num)
            for example in dataset:
                instance = extract_beavertails_harmful_response(example)
                if self.add_harmful_flag:
                    # Mark sample as harmful (poisoned)
                    instance["is_harmful"] = 1
                list_data_dict += [instance]

            normal_num = int((1 - poison_ratio) * sample_num)
            if normal_num > 0:
                benign_dataset = load_subset(benign_dataset, 0, normal_num)
                for sample in benign_dataset:
                    if self.add_harmful_flag:
                        # Mark sample as benign (safe)
                        sample["is_harmful"] = 0
                    list_data_dict += [sample]
            
            logger.debug(f"Benign data poisoned seed: {seed}")
            random.seed(seed)
            random.shuffle(list_data_dict)
        elif "advbench" in data_path:
            list_data_dict = []
            poison_num = int(poison_ratio * sample_num)
            dataset = load_subset("data/advbench.json", data_start_index, poison_num)
            for example in dataset:
                instance = extract_beavertails_harmful_response(example)
                if self.add_harmful_flag:
                    # Mark sample as harmful (poisoned)
                    instance["is_harmful"] = 1
                list_data_dict += [instance]

            normal_num = int((1 - poison_ratio) * sample_num)
            if normal_num > 0:
                benign_dataset = load_subset(benign_dataset, 0, normal_num)
                for sample in benign_dataset:
                    if self.add_harmful_flag:
                        # Mark sample as benign (safe)
                        sample["is_harmful"] = 0
                    list_data_dict += [sample]
            
            logger.debug(f"Benign data poisoned seed: {seed}")
            random.seed(seed)
            random.shuffle(list_data_dict)
        else:
            list_data_dict = jload(data_path)
            if self.add_harmful_flag:
                # Default to benign unless already specified in the JSON.
                logger.warning(f"Assume all samples are benign for {data_path}")
                for sample in list_data_dict:
                    sample.setdefault("is_harmful", 0)
        
        # NEW: If refusal responses are provided, create a *paired* refusal sample (same instruction/input, different output) for
        # each original item rather than appending them as independent training examples.
        refusal_targets = None
        if refusal_data_path is not None:
            from itertools import cycle, islice

            refusal_entries = jload_jsonl(refusal_data_path)
            if len(refusal_entries) == 0:
                logger.warning("Refusal jsonl file is empty – no refusal augmentation will be applied.")
            else:
                needed = len(list_data_dict)
                repeated_refusals = list(islice(cycle(refusal_entries), needed))
                # Build a list of refusal answers (string or dict) in the same order/length as list_data_dict.
                refusal_targets = list(repeated_refusals)
        
        logger.info("Formatting inputs...")
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        sources = [
            (
                prompt_input.format_map(example)
                if example.get("input", "") != ""
                else prompt_no_input.format_map(example)
            )
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]
        
        logger.info("Tokenizing inputs...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        # Persist the harmful indicator list for external reference (if enabled)
        self.is_harmful_flags = (
            [sample["is_harmful"] for sample in list_data_dict]
            if self.add_harmful_flag
            else None
        )

        # If refusal targets exist, tokenize them using the *same* sources but with the refusal answer.
        if refusal_targets is not None:
            refusal_data_dict = preprocess(sources, [f"{t}{tokenizer.eos_token}" for t in refusal_targets], tokenizer)
            self.input_ids_refusal = refusal_data_dict["input_ids"]
            self.labels_refusal = refusal_data_dict["labels"]
        else:
            # Placeholder lists so that attribute access is always defined.
            self.input_ids_refusal = None
            self.labels_refusal = None
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # When refusal samples exist, return both original and refusal tokenized pairs so that the collator can merge them
        if self.input_ids_refusal is not None:
            base_dict = dict(
                input_ids=self.input_ids[i],
                labels=self.labels[i],
                refusal_input_ids=self.input_ids_refusal[i],
                refusal_labels=self.labels_refusal[i],
            )
            if self.add_harmful_flag and self.is_harmful_flags is not None:
                base_dict["is_harmful"] = torch.tensor(self.is_harmful_flags[i])
            return base_dict
        # Fallback to original behaviour.
        base_dict = dict(input_ids=self.input_ids[i], labels=self.labels[i])
        if self.add_harmful_flag and self.is_harmful_flags is not None:
            base_dict["is_harmful"] = torch.tensor(self.is_harmful_flags[i])
        return base_dict


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(
        sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers and "lora" in name:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def load_subset(data_path: str, start_index: int, sample_num: int):
    """
    Return a subset of the dataset loaded with :func:`jload`.

    Parameters
    ----------
    data_path : str
        Path to the ``.json`` file to read.
    start_index : int
        Index of the first element to include in the subset.
    sample_num : int
        Number of samples to return. If 0, returns an empty list.
        If the requested range extends beyond the dataset length, 
        all available samples from ``start_index`` to the end are returned instead.

    Returns
    -------
    list
        A list containing the selected samples.
    """
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if sample_num is None or sample_num < 0:
        raise ValueError("sample_num must be a non-negative integer")
    
    # Handle the case where 0 samples are requested
    if sample_num == 0:
        logger.info(f"Returning 0 samples as requested (sample_num=0)")
        return []

    # Load the entire dataset first
    dataset = jload(data_path)

    # Compute the end index ensuring we don't go out of bounds
    end_index = min(len(dataset), start_index + sample_num)
    logger.info(f"start_index: {start_index}, end_index: {end_index}")
    subset = dataset[start_index:end_index]

    # Log the outcome
    logger.info(
        f"Returning {len(subset)} samples from {data_path} (start_index={start_index}, requested={sample_num})"
    )

    return subset
