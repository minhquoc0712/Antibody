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

from torch.nn import functional as F
from loguru import logger


class AntibodyAlignmentTrainer(Trainer):

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
        # Initialize exponential moving average for lambda_t
        self.lambda_t_ema = None
        self.ema_momentum = getattr(self.args, "ema_momentum", 0.9)
        # Check if wandb is initialized (skip when system evaluation mode)
        if (
            not getattr(self.args, "system_evaluate", False)
            and wandb.run is None
            and hasattr(self.args, "eta_lbd")
        ):
            lambda2_val = getattr(self.args, "lambda2", 0.1)
            wandb.init(
                project="Antibody",
                name=f"antibody_alignment_seed{self.args.seed}",
                config={
                    "alpha": self.args.alpha,
                    "eta_lbd": self.args.eta_lbd,
                    "lambda2": lambda2_val,
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
        # Augment with static hyper-parameters so they are recorded in every log entry.
        metrics_dict = dict(metrics_dict)  # work on a shallow copy to avoid side-effects.
        if hasattr(self.args, "alpha"):
            metrics_dict["alpha"] = self.args.alpha
        if hasattr(self.args, "eta_lbd"):
            metrics_dict["eta_lbd"] = self.args.eta_lbd

        # Respect system evaluation mode: suppress printing/logging when enabled
        if getattr(self.args, "system_evaluate", False):
            return

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
            # Newly added cosine similarities involving final gradients
            final_safe_cos_sim = self._compute_cosine_similarity(final_grads, safe_grads)
            final_diff_cos_sim = self._compute_cosine_similarity(final_grads, diff_grads)

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
                "final_safe_cos_sim": (
                    final_safe_cos_sim.item()
                    if isinstance(final_safe_cos_sim, torch.Tensor)
                    else final_safe_cos_sim
                ),
                "final_diff_cos_sim": (
                    final_diff_cos_sim.item()
                    if isinstance(final_diff_cos_sim, torch.Tensor)
                    else final_diff_cos_sim
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

    def _compute_lambda_t(self, alignment_grads, diff_grads):
        """Compute the lambda_t value for gradient scaling.

        Args:
            alignment_grads: Dictionary of gradients from the alignment loss.
            diff_grads: Dictionary of gradients representing the difference (e.g. harmful – perturbed).

        Returns:
            float: The computed raw lambda_t value
        """
        dot_product = 0.0
        diff_grad_norm = 0.0
        alignment_grad_norm = 0.0

        for name, align_grad in alignment_grads.items():
            if name not in diff_grads:
                raise ValueError(f"Parameter name '{name}' not found in diff_grads")

            # Flatten the gradients for dot-product and norm computations.
            align_grad_flat = align_grad.view(-1)
            diff_grad_flat = diff_grads[name].view(-1)

            dot_product += align_grad_flat.dot(diff_grad_flat)
            diff_grad_norm += diff_grad_flat.norm(2).item() ** 2
            alignment_grad_norm += align_grad_flat.norm(2).item() ** 2

        diff_grad_norm_square = diff_grad_norm
        alignment_grad_norm_square = alignment_grad_norm

        # model_grad_norm_square = alignment_grad_norm ** 0.5  # kept for reference (unused)
        # print(diff_grad_norm_square, dot_product)

        # TODO: Note that we change the code here.
        lambda_t = self.args.eta_lbd - (dot_product / (((diff_grad_norm ** 0.5) * (alignment_grad_norm ** 0.5)) + 1e-8))
        # lambda_t = self.args.eta_lbd - (dot_product / (diff_grad_norm_square + 1e-8))

        return max(lambda_t.detach(), 0.0)

    def _apply_lambda_t_ema(self, lambda_t_raw):
        """Apply exponential moving average to lambda_t value.

        Args:
            lambda_t_raw: The raw computed lambda_t value

        Returns:
            float: The smoothed lambda_t value using EMA
        """
        if self.lambda_t_ema is None:
            # First time: initialize EMA with the current value
            self.lambda_t_ema = lambda_t_raw
        else:
            # Update EMA: new_ema = momentum * old_ema + (1 - momentum) * current_value
            self.lambda_t_ema = self.ema_momentum * self.lambda_t_ema + (1 - self.ema_momentum) * lambda_t_raw

        return self.lambda_t_ema

    @torch.no_grad()
    def _sequence_nll(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        *,
        reduction: str = "sum",  # "sum" (default) or "mean"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-sequence negative log-likelihood and token counts.

        Parameters
        ----------
        model : nn.Module
        inputs : Dict[str, Tensor]

        Returns
        -------
        seq_nll : Tensor  shape (B,)
        token_counts : Tensor  shape (B,)
        """

        IGNORE_INDEX = -100

        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[..., :-1, :].contiguous()

        if "labels" in inputs:
            labels = inputs["labels"].clone()
        else:
            labels = inputs["input_ids"].clone()

        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        ce_flat = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )

        ce = ce_flat.view(shift_labels.size())  # (B, L-1)
        token_mask = shift_labels.ne(IGNORE_INDEX)

        seq_nll_sum = (ce * token_mask).sum(dim=1)  # (B,)

        token_counts = token_mask.sum(dim=1)    # (B,)

        if reduction == "mean":
            # Avoid division by zero (possible if all labels are IGNORE_INDEX)
            safe_counts = token_counts.clone()
            safe_counts[safe_counts == 0] = 1
            seq_nll = seq_nll_sum / safe_counts
        else:
            seq_nll = seq_nll_sum

        return seq_nll.detach(), token_counts.detach()

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        if "refusal_input_ids" in harmful_inputs:
            # Strip the "refusal_" prefix so the dict matches the model’s expected signature
            refusal_inputs = {
                k[len("refusal_"):]: v for k, v in harmful_inputs.items() if k.startswith("refusal_")
            }

            harmful_inputs = {k: v for k, v in harmful_inputs.items() if not k.startswith("refusal_")}
        else:
            refusal_inputs = None

        # ======================= Input Logging (Step 5 of Each Epoch) =======================
        # Skip heavy logging in system evaluation mode.
        if not getattr(self.args, "system_evaluate", False):
            # Centralised input logging: first advance the counter with safe inputs,
            # then (without advancing) log the corresponding harmful inputs so that
            # both views appear for the same batch when logging is enabled.
            self._maybe_log_inputs(inputs, dataset_type="safe", advance_counter=True)
            self._maybe_log_inputs(harmful_inputs, dataset_type="harmful", advance_counter=False)
            self._maybe_log_inputs(refusal_inputs, dataset_type="refusal", advance_counter=False)

        # ---------------- Harmful-flag visualization ----------------
        if (not getattr(self.args, "system_evaluate", False)) and ("is_harmful" in harmful_inputs):
            try:
                harm_flags_cpu = harmful_inputs["is_harmful"].detach().cpu()
                harmful_count = int((harm_flags_cpu == 1).sum().item())
                benign_count = int((harm_flags_cpu == 0).sum().item())
                logger.info(
                    f"[Batch {self.state.global_step}] is_harmful distribution – harmful: {harmful_count}, benign: {benign_count}, total: {harm_flags_cpu.numel()}"
                )
            except Exception as e:
                logger.warning(f"Failed to log harmful flag distribution: {e}")

        def step():
            # First compute and store the initial harmful gradient
            harmful_loss, harmful_grads = self.compute_loss_and_gradient(
                model, harmful_inputs
            )

            model_copy = copy.deepcopy(model)
            K = 1
            for k in range(K):
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model_copy, harmful_inputs)
                self.accelerator.backward(loss)
                temp_grads = {
                    name: param.grad.data.clone()
                    for name, param in model_copy.named_parameters()
                    if param.requires_grad
                }

                # Manual gradient update with projection
                lr = getattr(self.args, "alpha", 1e-1)
                with torch.no_grad():
                    grad_norm = self._grad_norm(temp_grads)

                    for param in model_copy.parameters():
                        if param.requires_grad and param.grad is not None:
                            param.data = param.data - lr * (
                                param.grad.data / (grad_norm + 1e-7)
                            )

                model_copy.zero_grad()

            # Get the final gradient after K steps
            perturbed_loss, perturb_grads = self.compute_loss_and_gradient(
                model_copy, harmful_inputs, zero_grad=True
            )
            diff_grads = {
                name: harmful_grads[name] - perturb_grads[name]
                for name in harmful_grads
            }

            # ---------------- Diff-CE gradient (refusal vs. harmful) ----------------
            diff_score_mean = None
            diff_ce_grads = None  # default when no refusal view
            if refusal_inputs is not None:
                # Compute gradients for refusal view
                refusal_loss_value, refusal_grads = self.compute_loss_and_gradient(
                    model_copy, refusal_inputs, zero_grad=True  # zero grads after storing
                )

                # Gradient of diff score (refusal - harmful)
                # diff_ce_grads = {
                #     name: refusal_grads[name] - perturb_grads[name]
                #     for name in refusal_grads if name in perturb_grads
                # }
                # diff_score_mean = refusal_loss_value - perturbed_loss
                diff_ce_grads = refusal_grads
                diff_score_mean = refusal_loss_value

            del model_copy
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Calculate the alignment gradient
            alignment_safe_loss, safe_grads = self.compute_loss_and_gradient(
                model, inputs, zero_grad=False
            )
            lambda_t_raw = self._compute_lambda_t(alignment_grads=safe_grads, diff_grads=diff_grads)
            lambda_t = self._apply_lambda_t_ema(lambda_t_raw)
            lambda2 = self.args.lambda2

            if not getattr(self.args, "system_evaluate", False):
                print(self.args.system_evaluate)
                logger.debug(f"lambda_t_raw: {lambda_t_raw}, lambda_t_ema: {lambda_t}, lambda2: {lambda2}, ema_momentum: {self.ema_momentum}")
            

            # Apply final gradient
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                base_grad = safe_grads[name] + lambda_t * (
                    harmful_grads[name] - perturb_grads[name]
                )

                # Add diff-CE gradient if available
                if diff_ce_grads is not None and name in diff_ce_grads:
                    base_grad = base_grad + lambda2 * diff_ce_grads[name]

                param.grad.data = base_grad

            self.steps += 1

            # Compute and log metrics only every 10 steps (skip in system evaluation mode)
            if (self.steps % 10 == 0) and (not getattr(self.args, "system_evaluate", False)):
                final_grads = {
                    name: param.grad.data.clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }

                # Compute all metrics
                metrics_dict = self._compute_metrics(
                    harmful_grads=harmful_grads,
                    perturb_grads=perturb_grads,
                    safe_grads=safe_grads,
                    diff_grads=diff_grads,
                    final_grads=final_grads,
                    harmful_loss=harmful_loss,
                    perturbed_loss=perturbed_loss,
                    alignment_safe_loss=alignment_safe_loss,
                )

                # Include current lambda_t in the metrics that get logged (e.g. to wandb)
                metrics_dict["lambda_t"] = lambda_t

                # ----- Append CE statistics if available -----
                if diff_score_mean is not None:
                    metrics_dict["refusal_loss"] = refusal_loss_value
                    metrics_dict["diff_score"] = diff_score_mean
                    metrics_dict["lambda2"] = lambda2

                # Log all metrics
                self._log_metrics(metrics_dict)

            # Integrate diff score into the scalar loss for logging purposes
            diff_loss_term = diff_score_mean if diff_score_mean is not None else 0.0

            final_loss = (
                alignment_safe_loss
                + lambda_t * (harmful_loss - perturbed_loss)
                + diff_loss_term
            )
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

        # Skip any input logging in system evaluation mode
        if getattr(self.args, "system_evaluate", False):
            return

        try:
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
