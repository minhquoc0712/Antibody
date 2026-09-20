<!-- markdownlint-disable first-line-h1 -->

<!-- markdownlint-disable html -->

<h1 align="center">Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence</h1>

This repository contains the implementation of [Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence (ICLR 2026)](https://openreview.net/pdf?id=qur2ef8MqQ).

## Overview

Fine-tuning-as-a-service introduces a threat to Large Language Models’ safety when service providers fine-tune their models on poisoned user-submitted datasets, a process known as harmful fine-tuning attacks. In this work, we show that by regularizing the gradient contribution of harmful samples encountered during finetuning, we can effectively mitigate the impact of harmful fine-tuning attacks. To this end, we introduce Antibody, a defense strategy that first ensures robust safety alignment for the model before fine-tuning, and then applies a safety-preservation learning algorithm during fine-tuning. Specifically, in the alignment stage before fine-tuning, we propose optimizing the model to be in a flat loss region with respect to harmful samples, which makes the safety alignment more resilient to subsequent harmful fine-tuning. Then, in the fine-tuning stage, we design a fine-tuning algorithm that applies a weighting scheme to all samples in each training batch to inhibit the model from learning from harmful samples while encouraging learning from benign samples. Experimental results demonstrate that Antibody successfully mitigates harmful fine-tuning attacks while boosting fine-tuning performance on the user-submitted dataset.

## Setup

Use your system’s Conda installation.

```bash
cd /absolute/path/to/Antibody
export ANTIBODY_ROOT="$PWD"
mkdir -p cache ckpt logs
conda env create -f environment.yml
conda activate adaptive-attack
```

## Model Access

Obtain access to [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). Save your Hugging Face read token in `huggingface_token.txt`.  Download the base model and safety evaluator:

```python
from pathlib import Path
from huggingface_hub import login, snapshot_download

token = Path("huggingface_token.txt").read_text().strip()
login(token=token, add_to_git_credential=False)
cache = str(Path("cache").resolve())
model = snapshot_download(
    "meta-llama/Llama-2-7b-hf",
    revision="01c7f73d771dfac7d292323805ebc428287df4f9",
    cache_dir=cache, token=token,
)
snapshot_download("PKU-Alignment/beaver-dam-7b", cache_dir=cache, token=token)
print('model_path="' + model + '"')
```

In [the main script](script/antibody/antibody_alignment_finetune.sh):

- In each script you run, replace the placeholder `model_path` with the absolute path to your downloaded model. For the main Llama-2 script, use the printed path above.
- Adjust `module load miniforge3` and `module load cuda/12.0` for your cluster. Remove those two lines if your machine does not use modules.
- Ensure `CUDA_VISIBLE_DEVICES=0` selects your allocated GPU.

## Data Preparation

The processed alignment, harmful fine-tuning, safety evaluation, and refusal data are included in `data/`. Keep these four files unchanged:

- `beavertails_with_refusals_train_filtered.json`
- `beavertails_disjoint_attack_deduplicated.json`
- `beavertails_evaluation.json`
- `refusal_examples.jsonl`

Following [Booster's data preparation](https://github.com/git-disl/Booster#data--preparation), generate the supervised fine-tuning data:

```bash
cd sst2 && python build_dataset.py
cd agnews && python build_dataset.py
cd gsm8k && python build_dataset.py
```

These commands download the training splits and create `data/sst2.json`, `data/agnews.json`, and `data/gsm8k.json`. These generated files are excluded by `.gitignore`. The evaluators download their evaluation splits automatically.

## Logging and Verification

Log in to W&amp;B:

```bash
wandb login
```

The script sets `WANDB_MODE="online"`. Keep network access available for Hugging Face and W&amp;B.

## Run Antibody

Start from `script/antibody/`:

```bash
cd "$ANTIBODY_ROOT/script/antibody"
bash ./antibody_alignment_finetune.sh
```

Both stages use 20 epochs and batch size 16. Learning rates are `5e-4` for alignment and `1e-5` for fine-tuning. Each task uses 1000 fine-tuning samples. These settings are defined inside the script.

The experiments need one A100-80GB or H100.

## Outputs

- `ckpt/`: alignment and task-specific LoRA adapters.
- `data/poison/`: safety predictions and `*_sentiment_eval.json` scores.
- `data/sst2/`, `data/agnews/`, `data/gsm8k/`: task predictions and accuracy.
- `wandb/`: experiment logs.

A complete run finishes GSM8K evaluation and exits successfully. Outputs use the Slurm job ID, or `joblocal` outside Slurm. Use a fresh working copy for each local run to avoid overwriting results.

## Acknowledgements

The code is built on[Booster](https://github.com/git-disl/Booster).

## Citation

```
@inproceedings{nguyen2026antibody,
    title={Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence},
    author={Quoc Minh Nguyen and Trung Le and Jing Wu and Anh Tuan Bui and Mehrtash Harandi},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=qur2ef8MqQ}
} 
```

