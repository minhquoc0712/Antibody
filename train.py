#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers
import random
import numpy as np
import torch
import transformers
from transformers import TrainerCallback
from torch.utils.data import Dataset
from trainers import (
    SFTTrainer,
    WeightedSFTTrainer,
    BoosterAlignmentTrainer,
    LisaTrainer,
    RepNoiseTrainer,
    AntibodyAlignmentTrainer,
    VaccineTrainer,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import json
import wandb
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from callbacks import GPUTimeCallback, GPUMemoryCallback, GPUPeakMemoryCallback
from loguru import logger

sys.path.append("..")
import utils
from utils import SupervisedDataset

# // Set access token (NB: Keep this private!)
access_token = next(open("huggingface_token.txt")).strip()


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def set_seed(seed):
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Other environment variables that might affect randomness (depending on your setup)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning. Supports paired (original + refusal) instances."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        paired = "refusal_input_ids" in instances[0]

        # Stack the per-sample harm flags into a single tensor so downstream code can inspect it.
        harms = torch.stack([inst["is_harmful"] for inst in instances]) if "is_harmful" in instances[0] else None

        if paired:
            input_ids_orig = [inst["input_ids"] for inst in instances]
            labels_orig = [inst["labels"] for inst in instances]

            input_ids_ref = [inst["refusal_input_ids"] for inst in instances]
            labels_ref = [inst["refusal_labels"] for inst in instances]

            # Pad each set independently so their sequence lengths don't constrain each other
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_orig,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_orig, batch_first=True, padding_value=IGNORE_INDEX
            )

            refusal_input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_ref,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            refusal_labels = torch.nn.utils.rnn.pad_sequence(
                labels_ref, batch_first=True, padding_value=IGNORE_INDEX
            )

            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
                "refusal_input_ids": refusal_input_ids,
                "refusal_labels": refusal_labels,
                "refusal_attention_mask": refusal_input_ids.ne(
                    self.tokenizer.pad_token_id
                ),
                # Pass the harm indicator along with the batch (same for both safe and refusal views).
                **({"is_harmful": harms} if harms is not None else {}),
            }
        else:
            input_ids_list, labels_list = tuple(
                [instance[key] for instance in instances]
                for key in ("input_ids", "labels")
            )

            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=IGNORE_INDEX
            )
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                **({"is_harmful": harms} if harms is not None else {}),
            )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, training_args, refusal_data_path=None
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    if "beavertails_with_refusals_train_filtered_safe" in data_args.data_path:
        train_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poison_ratio=0.0,
            sample_num=int(data_args.sample_num),
            benign_dataset=None,
            data_start_index=0,
            seed=training_args.seed,
        )
    elif "beavertails_with_refusals_train_filtered_harmful" in data_args.data_path:
        train_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poison_ratio=1.0,
            sample_num=int(data_args.sample_num),
            benign_dataset=None,
            data_start_index=0,
            seed=training_args.seed,
        )
    elif "beavertails_disjoint_attack_deduplicated" in data_args.data_path:
        train_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poison_ratio=data_args.poison_ratio,
            sample_num=int(data_args.sample_num),
            benign_dataset=data_args.benign_dataset,
            data_start_index=0,
            seed=training_args.seed,
            refusal_data_path=refusal_data_path,
            add_harmful_flag=True,
        )
    elif "advbench" in data_args.data_path:
        train_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            poison_ratio=data_args.poison_ratio,
            sample_num=int(data_args.sample_num),
            benign_dataset=data_args.benign_dataset,
            data_start_index=0,
            seed=training_args.seed,
            refusal_data_path=refusal_data_path,
            add_harmful_flag=True,
        )
    else:
        raise ValueError(f"Invalid data path: {data_args.data_path}")

    alignment_safe_eval_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path="beavertails_with_refusals_train_filtered_safe",
        poison_ratio=0.0,
        sample_num=200,
        benign_dataset=None,
        data_start_index=0,
        seed=training_args.seed,
    )

    alignment_harmful_eval_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path="beavertails_with_refusals_train_filtered_harmful",
        poison_ratio=1.0,
        sample_num=200,
        benign_dataset=data_args.benign_dataset,
        data_start_index=0,
        seed=training_args.seed,
    )

    finetuning_harmful_eval_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path="beavertails_disjoint_attack_deduplicated",
        poison_ratio=1.0,
        sample_num=200,
        benign_dataset=data_args.benign_dataset,
        data_start_index=2000,
        seed=training_args.seed,
    )

    
    # # Pure harmful dataset - contains only the harmful samples from training
    # pure_harmful_eval_dataset = SupervisedDataset(
    #     tokenizer=tokenizer,
    #     data_path=data_args.data_path,  # Same path as training
    #     poison_ratio=1.0,  # Only harmful samples
    #     sample_num=int(data_args.poison_ratio * data_args.sample_num),
    #     benign_dataset=data_args.benign_dataset,
    #     data_start_index=0,  # Same start index as training
    #     seed=training_args.seed,  # Same seed as training
    #     refusal_data_path=None,
    #     add_harmful_flag=False,
    # )
    
    # # Pure benign dataset - contains only the benign samples from training
    # pure_benign_eval_dataset = SupervisedDataset(
    #     tokenizer=tokenizer,
    #     data_path=data_args.data_path,
    #     poison_ratio=0.0,  # Only benign samples
    #     sample_num=int((1 - data_args.poison_ratio) * data_args.sample_num),
    #     benign_dataset=data_args.benign_dataset,
    #     data_start_index=0,  # Start from beginning of benign dataset (same as training logic)
    #     seed=training_args.seed,  # Same seed as training
    #     refusal_data_path=None,
    #     add_harmful_flag=False,
    # )

    # logger.debug(f"Pure harmful length: {len(pure_harmful_eval_dataset)}")
    # logger.debug(f"Pure benign length: {len(pure_benign_eval_dataset)}")

    eval_dataset = {
        "alignment_safe": alignment_safe_eval_dataset,
        "alignment_harmful": alignment_harmful_eval_dataset,
        "finetuning_harmful": finetuning_harmful_eval_dataset,
        # "pure_harmful": pure_harmful_eval_dataset,
        # "pure_benign": pure_benign_eval_dataset,
    }

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def log_hyperparameters(args_dict):
    """Log hyperparameters in a beautiful table format using rich."""
    console = Console()

    # Create a table
    table = Table(
        title="Training Hyperparameters", show_header=True, header_style="bold magenta"
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    # Add rows for each hyperparameter
    for key, value in args_dict.items():
        if isinstance(value, (int, float, str, bool)):
            table.add_row(str(key), str(value))

    # Print the table
    console.print(table)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    # Add wandb arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="antibody",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Weights & Biases entity name"
    )

    parser.add_argument(
        "--optimizer", type=str, default="AdamW", help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--lora_folder", type=str, default="", help="Specify the lora path"
    )
    parser.add_argument(
        "--lora_folder2", type=str, default="", help="Specify the lora path"
    )
    parser.add_argument(
        "--rho", type=float, default=0.1, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--poison_ratio", type=float, default=0.1, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--sample_num", type=float, default=5000, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--benign_dataset",
        type=str,
        default="data/sst2.json",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--vaccine_ratio", type=float, default=0, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--lamb", type=float, default=0.001, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--epsilon", type=float, default=5e-4, help="Epsilon for gradient projection"
    )
    parser.add_argument(
        "--track_embedding_before_train",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--track_embedding_drift",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--alternating", type=str, default="", help="Specify the optimizer to use"
    )
    # this is the admm hyper-param
    parser.add_argument(
        "--finetune_step", type=int, default=500, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--alignment_step", type=int, default=500, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--guide_data_num", type=int, default=10000, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--dense_ratio", type=float, default=0.1, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--noise_variance", type=float, default=0.1, help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--bad_sample_num",
        type=float,
        default=1000,
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--good_sample_num",
        type=float,
        default=1000,
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--system_evaluate",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--no_harmful_dataset",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--no_safety_mask",
        type=str,
        default="True",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--random_prune", type=str, default="False", help="Specify the optimizer to use"
    )
    parser.add_argument(
        "--full_model_prune",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--perturb_aware",
        type=str,
        default="False",
        help="Specify the optimizer to use",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Learning rate for safe gradients"
    )
    parser.add_argument(
        "--num_ascent_steps",
        type=int,
        default=1,
        help="Number of gradient ascent steps in inner loop",
    )
    parser.add_argument(
        "--eta_lbd", type=float, default=0.1, help="Lambda update rate in inner loop"
    )
    # Temperature (tau) for softmax weighting in WeightedSFT
    parser.add_argument(
        "--tau", type=float, default=1.0, help="Temperature for softmax weighting in WeightedSFT"
    )
    parser.add_argument(
        "--lambda2", type=float, default=0.1, help="Refusal-loss coefficient for Antibody alignment"
    )
    parser.add_argument(
        "--harmful_batch_size",
        type=int,
        default=None,
        help="Batch size for harmful data loading. Defaults to per_device_train_batch_size if not specified",
    )
    parser.add_argument(
        "--ema_momentum",
        type=float,
        default=0.9,
        help="Exponential moving average momentum parameter for lambda_t smoothing in AntibodyAlignmentTrainer",
    )

    model_args, data_args, training_args, extra_args = (
        parser.parse_args_into_dataclasses()
    )
    args = parser.parse_args()
    training_args.optimizer = extra_args.optimizer
    training_args.rho = extra_args.rho
    training_args.lamb = extra_args.lamb
    training_args.track_embedding_before_train = extra_args.track_embedding_before_train
    training_args.alternating = extra_args.alternating
    training_args.epsilon = extra_args.epsilon
    data_args.poison_ratio = extra_args.poison_ratio
    data_args.sample_num = extra_args.sample_num
    data_args.benign_dataset = extra_args.benign_dataset
    data_args.vaccine_ratio = extra_args.vaccine_ratio
    data_args.guide_data_num = extra_args.guide_data_num
    data_args.bad_sample_num = extra_args.bad_sample_num
    data_args.good_sample_num = extra_args.good_sample_num
    training_args.guide_data_num = extra_args.guide_data_num
    training_args.rho = extra_args.rho
    training_args.finetune_step = extra_args.finetune_step
    training_args.alignment_step = extra_args.alignment_step
    training_args.dense_ratio = extra_args.dense_ratio
    training_args.noise_variance = extra_args.noise_variance
    training_args.model = model_args.model_name_or_path
    training_args.track_embedding_drift = extra_args.track_embedding_drift
    training_args.system_evaluate = extra_args.system_evaluate
    training_args.no_harmful_dataset = extra_args.no_harmful_dataset
    training_args.no_safety_mask = extra_args.no_safety_mask
    training_args.random_prune = extra_args.random_prune
    training_args.full_model_prune = extra_args.full_model_prune
    training_args.sample_num = extra_args.sample_num
    training_args.alpha = extra_args.alpha
    training_args.model_max_length = 256

    # ======= Trainer arguments =======
    training_args.num_ascent_steps = extra_args.num_ascent_steps
    training_args.eta_lbd = extra_args.eta_lbd
    # Propagate tau hyper-parameter so trainers can access it via self.args.tau
    training_args.tau = extra_args.tau
    # Propagate lambda2 so trainers can access it via self.args.lambda2
    training_args.lambda2 = extra_args.lambda2
    training_args.lamb = extra_args.lamb
    training_args.harmful_batch_size = extra_args.harmful_batch_size
    # Propagate ema_momentum so trainers can access it via self.args.ema_momentum
    training_args.ema_momentum = extra_args.ema_momentum
    # ====================================

    seed = training_args.seed
    logger.debug("seed {}".format(seed))
    set_seed(seed)
    # if "gemma" in model_args.model_name_or_path or "Mistral" in model_args.model_name_or_path:
    #     # to prevent oom
    #     training_args.model_max_length=180

    training_args.perturb_aware = extra_args.perturb_aware
    # if data_args.benign_dataset== "data/alpaca.json":
    #     # to prevent oom
    #     training_args.model_max_length=512

    if extra_args.optimizer == "rep_noise":
        # to prevent oom
        training_args.model_max_length = 256

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        cache_dir=training_args.cache_dir,
        device_map="auto",
        token=access_token,
    )

    logger.debug(f"Model name: {model_args.model_name_or_path}")
    if "gemma" in model_args.model_name_or_path:
        logger.debug("Loading gemma model with eager attn")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=False,
            cache_dir=training_args.cache_dir,
            device_map="auto",
            token=access_token,
            attn_implementation='eager',
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_fast=True,
        padding_side="right",
        model_max_length=training_args.model_max_length,
        token=access_token,
    )

    # Enable BF16 precision
    model = model.to(torch.bfloat16)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    logger.info(f"Tokenizer length: {len(tokenizer)}")

    if training_args.optimizer == "EWC" or training_args.alternating == "single_lora":
        first_lora_trainable = True
        logger.info("single_lora here !!!!!!!")
    else:
        first_lora_trainable = False

    if extra_args.lora_folder != "":
        logger.info("Recover LoRA weights..")
        model = PeftModel.from_pretrained(
            model, extra_args.lora_folder, is_trainable=first_lora_trainable
        )
        # single lora method don't need to merge and load second lora

        if not first_lora_trainable:
            model = model.merge_and_unload()
            if extra_args.lora_folder2 == "":
                logger.info("Creating new second lora for training")
                config = LoraConfig(
                    r=32,
                    lora_alpha=4,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    lora_dropout=0,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model = get_peft_model(model, config)
            else:
                # load second lora and used for training
                model = PeftModel.from_pretrained(
                    model, extra_args.lora_folder2, is_trainable=True
                )
                logger.info(model.peft_config)
    else:
        # create first lora
        logger.info("Initialize Lora weights..")
        config = LoraConfig(
            r=32,
            lora_alpha=4,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    model.print_trainable_parameters()
    # logger.info(training_args)
    model.train()

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )
    harmful_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path="beavertails_with_refusals_train_filtered_harmful",
        poison_ratio=1.0,
        sample_num=int(data_args.bad_sample_num),
        benign_dataset=None,
        data_start_index=0,
        seed=training_args.seed,
        refusal_data_path="./data/refusal_examples.jsonl" if training_args.optimizer == "antibody_alignment" else None,
    )
    if training_args.optimizer == "rep_noise":
        trainer = RepNoiseTrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
        trainer.init(harmful_dataset)
    elif training_args.optimizer=="vaccine":
        trainer = VaccineTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
    # elif training_args.optimizer == "random_vaccine":
    #     trainer = RandomVaccineTrainer(model=model, tokenizer=tokenizer, args=training_args,**data_module)
    elif training_args.optimizer == "lisa":
        trainer = LisaTrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
        # alignment_dataset = SupervisedDataset(
        #     tokenizer=tokenizer,
        #     data_path="BeaverTails_safe",
        #     sample_num=data_args.guide_data_num,
        #     seed=training_args.seed,
        # )
        alignment_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path="beavertails_with_refusals_train_filtered_safe",
            poison_ratio=0.0,
            sample_num=int(data_args.guide_data_num),
            benign_dataset=None,
            data_start_index=0,
            seed=training_args.seed,
        )
        trainer.init(alignment_dataset)
    elif training_args.optimizer == "booster":
        trainer = BoosterAlignmentTrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
        trainer.init(harmful_dataset)
    elif training_args.optimizer == "antibody_alignment":
        logger.debug(f"Training with antibody_alignment")
        logger.debug(f"training_args.remove_unused_columns: {training_args.remove_unused_columns}")
        trainer = AntibodyAlignmentTrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
        trainer.init(harmful_dataset)
    elif training_args.optimizer == "sft":
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
        trainer.init(harmful_dataset)
    elif training_args.optimizer == "weighted_sft":
        logger.debug(f"Training with weighted_sft")
        data_module = make_supervised_data_module(
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
            refusal_data_path="./data/refusal_examples.jsonl",
        )
        logger.debug(f"train_dataset: {data_module['train_dataset'][0].keys()}")
        training_args.remove_unused_columns = False
        trainer = WeightedSFTTrainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )
        trainer.init(harmful_dataset)
    else:
        logger.info("Using normal finetuning...")
        trainer = transformers.Trainer(
            model=model, tokenizer=tokenizer, args=training_args, **data_module
        )

    # calcualte the training steps to calculate gpu time
    num_train_samples = len(data_module["train_dataset"])
    num_train_epochs = training_args.num_train_epochs
    train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    effective_batch_size = train_batch_size * gradient_accumulation_steps
    total_steps = num_train_epochs * (num_train_samples // effective_batch_size)

    if training_args.system_evaluate == "True":
        trainer.add_callback(GPUTimeCallback(total_steps))
        trainer.add_callback(GPUMemoryCallback())
        trainer.add_callback(GPUPeakMemoryCallback())
        # trainer.add_callback(EmbeddingCallback())

    # Add wandb callback for normal finetuning
    if training_args.optimizer == "normal":
        # Create a descriptive run name
        run_name = f"{training_args.optimizer}"
        if extra_args.lora_folder:
            run_name += f"_lora_{extra_args.lora_folder.split('/')[-1]}"
        if data_args.benign_dataset:
            run_name += (
                f"_data_{data_args.benign_dataset.split('/')[-1].replace('.json', '')}"
            )
        run_name += f"_poison{data_args.poison_ratio}_samples{data_args.sample_num}"

    if training_args.num_train_epochs > 0:
        logger.info(f"Training for {total_steps} steps")
        trainer.train()

    trainer.save_state()
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
