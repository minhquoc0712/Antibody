from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import time
import torch
import collections
import copy
import random
from packaging import version
import torch.nn as nn
from transformers import Trainer
from transformers import logging
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, RandomSampler
import wandb
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
)
from torch.utils.data import DataLoader, RandomSampler

from .logging_utils import log_input_data_details

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

# Import apex amp for mixed precision training
try:
    from apex import amp
except ImportError:
    amp = None

logger = logging.get_logger(__name__)


class SFTTrainer(Trainer):

    def get_harmful_dataloader(self, harmful_datast) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        data_collator = self.data_collator

        # Create a generator with seed for reproducible sampling
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        sampler = RandomSampler(harmful_datast, generator=generator)

        harmful_batch_size = self.args.per_device_train_batch_size
        logger.debug(f"Harmful batch size: {harmful_batch_size}")

        dataloader_params = {
            "batch_size": harmful_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmful_datast, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmful_datast, **dataloader_params))

    def init(self, harmful_datast):
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num > 0:
            self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
            self.harmful_data_iter = iter(self.harmful_dataloader)
        self.statistic = 0
        # Respect system evaluation mode: disable any W&B initialisation
        if (
            self.args.system_evaluate != "True"
            and wandb.run is None
            and hasattr(self.args, "eta_lbd")
        ):
            wandb.init(
                project="SFT",
                name=f"sft",
                config={
                    "alpha": self.args.alpha,
                    "eta_lbd": self.args.eta_lbd,
                    "num_ascent_steps": self.args.num_ascent_steps,
                },
            )

    def sample_from_harmful(self):
        # Get a  batch
        try:
            batch = next(self.harmful_data_iter)
        except StopIteration:
            # If the iterator is exhausted, create a new iterator
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch

    @torch.no_grad()
    def _grad_norm(self, stored_grad):
        grad_norm = 0.0
        for name, grad in stored_grad.items():
            grad_norm += grad.norm(2) ** 2
        return grad_norm ** 0.5

    def _compute_cosine_similarity(self, grad_dict1, grad_dict2):
        """
        Compute cosine similarity between two dictionaries of gradients.
        """
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for name in grad_dict1:
            if name in grad_dict2:
                grad1 = grad_dict1[name].view(-1)
                grad2 = grad_dict2[name].view(-1)

                dot_product += grad1.dot(grad2)
                norm1 += grad1.norm(2) ** 2
                norm2 += grad2.norm(2) ** 2

        return dot_product / (torch.sqrt(norm1 * norm2) + 1e-8)

    def _log_metrics(self, metrics_dict):
        """Centralized method for logging all metrics.

        Args:
            metrics_dict: Dictionary containing all metrics to log
        """
        # Skip all logging in system evaluation mode
        if self.args.system_evaluate == "True":
            return
        # Augment with static hyper-parameters so they are recorded in every log entry.
        metrics_dict = dict(metrics_dict)  # work on a shallow copy to avoid side-effects.
        if hasattr(self.args, "alpha"):
            metrics_dict["alpha"] = self.args.alpha
        if hasattr(self.args, "eta_lbd"):
            metrics_dict["eta_lbd"] = self.args.eta_lbd

        # Print metrics to console only every 10 steps
        print(f"\n===== Step {self.steps} Metrics =====", flush=True)
        for key, value in metrics_dict.items():
            if key != "step":  # Skip step in console output
                print(f"{key}: {value}", flush=True)

        # Log metrics to wandb on every step
        if wandb.run is not None:
            wandb.log(metrics_dict)

    def _compute_metrics(
        self,
        harmful_grads,
        perturb_grads,
        safe_grads,
        diff_grads,
        final_grads,
        harmful_loss,
        perturbed_loss,
        alignment_safe_loss,
    ):
        """
        Compute all metrics for logging.
        """
        with torch.no_grad():
            harmful_grad_norm = self._grad_norm(harmful_grads) + 1e-7
            perturbed_grad_norm = self._grad_norm(perturb_grads) + 1e-7
            safe_grad_norm = self._grad_norm(safe_grads) + 1e-7
            diff_grad_norm = self._grad_norm(diff_grads)
            final_grad_norm = self._grad_norm(final_grads) + 1e-7

            # Compute cosine similarities
            safe_harmful_cos_sim = self._compute_cosine_similarity(
                safe_grads, harmful_grads
            )
            safe_perturbed_cos_sim = self._compute_cosine_similarity(
                safe_grads, perturb_grads
            )
            safe_diff_cos_sim = self._compute_cosine_similarity(safe_grads, diff_grads)
            harmful_perturbed_cos_sim = self._compute_cosine_similarity(
                harmful_grads, perturb_grads
            )

            # Create metrics dictionary
            metrics_dict = {
                # Losses
                "harmful_loss": harmful_loss,
                "perturbed_loss": perturbed_loss,
                "loss_change": (harmful_loss - perturbed_loss),
                "alignment_safe_loss": alignment_safe_loss,
                # Gradient norms
                "harmful_grad_norm": (
                    harmful_grad_norm.item()
                    if isinstance(harmful_grad_norm, torch.Tensor)
                    else harmful_grad_norm
                ),
                "perturbed_grad_norm": (
                    perturbed_grad_norm.item()
                    if isinstance(perturbed_grad_norm, torch.Tensor)
                    else perturbed_grad_norm
                ),
                "safe_grad_norm": (
                    safe_grad_norm.item()
                    if isinstance(safe_grad_norm, torch.Tensor)
                    else safe_grad_norm
                ),
                "diff_grad_norm": (
                    diff_grad_norm.item()
                    if isinstance(diff_grad_norm, torch.Tensor)
                    else diff_grad_norm
                ),
                "final_grad_norm": (
                    final_grad_norm.item()
                    if isinstance(final_grad_norm, torch.Tensor)
                    else final_grad_norm
                ),
                # Cosine similarities
                "safe_harmful_cos_sim": (
                    safe_harmful_cos_sim.item()
                    if isinstance(safe_harmful_cos_sim, torch.Tensor)
                    else safe_harmful_cos_sim
                ),
                "safe_perturbed_cos_sim": (
                    safe_perturbed_cos_sim.item()
                    if isinstance(safe_perturbed_cos_sim, torch.Tensor)
                    else safe_perturbed_cos_sim
                ),
                "safe_diff_cos_sim": (
                    safe_diff_cos_sim.item()
                    if isinstance(safe_diff_cos_sim, torch.Tensor)
                    else safe_diff_cos_sim
                ),
                "harmful_perturbed_cos_sim": (
                    harmful_perturbed_cos_sim.item()
                    if isinstance(harmful_perturbed_cos_sim, torch.Tensor)
                    else harmful_perturbed_cos_sim
                ),
                "step": self.steps,
            }

            return metrics_dict

    def compute_loss_and_gradient(self, model, inputs, zero_grad=True):
        """Compute gradients for given model and inputs.

        Args:
            model: The model to compute gradients for
            inputs: Input batch
            zero_grad: Whether to zero gradients after computation

        Returns:
            Tuple[float, Dict[str, torch.Tensor]]: Loss value and gradients dictionary
        """
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            loss_value = loss.item()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        # Store gradients
        gradients = {
            name: param.grad.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        if zero_grad:
            model.zero_grad()

        return loss_value, gradients

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)

        # ======================= Input Logging (Step 5 of Each Epoch) =======================
        # Centralised input logging: first advance the counter with safe inputs,
        # then (without advancing) log the corresponding harmful inputs so that
        # both views appear for the same batch when logging is enabled.
        self._maybe_log_inputs(inputs, dataset_type="safe", advance_counter=True)

        def step():
            # Calculate the alignment gradient
            alignment_safe_loss, safe_grads = self.compute_loss_and_gradient(
                model, inputs, zero_grad=False
            )

            # Apply final gradient
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.grad.data = safe_grads[name]
                    

            self.steps += 1

            # Compute and log metrics only every 10 steps
            if self.steps % 10 == 0:
                final_grads = {
                    name: param.grad.data.clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                final_grad_norm = self._grad_norm(final_grads) + 1e-7
                metrics_dict = {
                    "step": self.steps,
                    "final_grad_norm": final_grad_norm.item(),
                    "alignment_safe_loss": alignment_safe_loss,
                }

            #     # Compute all metrics
            #     metrics_dict = self._compute_metrics(
            #         safe_grads=safe_grads,
            #         final_grads=final_grads,
            #         alignment_safe_loss=alignment_safe_loss,
            #     )

                # Log all metrics
                self._log_metrics(metrics_dict)

            final_loss = alignment_safe_loss
            return torch.tensor(final_loss, requires_grad=True, device=self.args.device)

        loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps

    def _maybe_log_inputs(self, inputs, dataset_type: str = "single", *, advance_counter: bool = True):
        """Log a sample of the current input batch once per epoch (step 5).

        This method centralises the logic for deciding *when* and *how* to log
        a detokenised view of the incoming batch using the helper function
        ``log_input_data_details`` defined in ``trainers.logging_utils``.

        It assumes that ``self.tokenizer`` is available (the standard
        ``transformers.Trainer`` already stores the tokenizer when provided)
        and relies on ``self.state`` for epoch / global-step tracking.
        """

        try:
            # Skip input logging entirely in system evaluation mode
            if self.args.system_evaluate == "True":
                return
            # Current epoch may be a float when using fractional progress.
            current_epoch_idx = int(self.state.epoch) if self.state.epoch is not None else 0

            # Initialise tracking attributes on first call.
            if not hasattr(self, "_log_epoch_idx"):
                self._log_epoch_idx = current_epoch_idx
                self._log_step_in_epoch = 0

            # Reset per-epoch counter on new epoch.
            if current_epoch_idx != self._log_epoch_idx:
                self._log_epoch_idx = current_epoch_idx
                self._log_step_in_epoch = 0

            # Optionally advance the per-epoch step counter (only once per batch).
            if advance_counter:
                self._log_step_in_epoch += 1

            # Determine whether we should log for this batch.
            should_log = getattr(self, "_should_log_current_batch", False)
            if advance_counter:
                should_log = self._log_step_in_epoch == 5
                self._should_log_current_batch = should_log

            if not should_log:
                return  # Nothing to log this call

            # Extract input_ids for a single example to detokenise.
            if "input_ids" not in inputs:
                raise ValueError("input_ids not found in inputs")

            sample_input_ids = inputs["input_ids"][0]

            # Ensure that we have both a sample and a tokenizer.
            if sample_input_ids is None or getattr(self, "tokenizer", None) is None:
                return

            # Perform the actual logging via utility.
            log_input_data_details(
                sample_input_ids=sample_input_ids,
                inputs=inputs,
                tokenizer=self.tokenizer,
                current_epoch=current_epoch_idx,
                step_in_epoch=self._log_step_in_epoch,
                global_step=self.state.global_step,
                dataset_type=dataset_type,
            )

        except Exception as e:
            # Protect training from any logging failures.
            logger.warning(
                f"Input logging failed at epoch {self.state.epoch}, global step {self.state.global_step}: {e}"
            )