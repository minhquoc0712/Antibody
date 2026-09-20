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
import torch.nn.functional as F  # NEW: for manual NLL computation
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

from loguru import logger


class WeightedSFTTrainer(Trainer):

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
        # Check if wandb is initialized (skip when system evaluation mode)
        if (
            not getattr(self.args, "system_evaluate", False)
            and wandb.run is None
            and hasattr(self.args, "eta_lbd")
        ):
            wandb.init(
                project="WeightedSFT",
                name=f"weighted_sft",
                config={
                    "alpha": self.args.alpha,
                    "eta_lbd": self.args.eta_lbd,
                    "tau": getattr(self.args, "tau", 1.0),
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

    # ------------------------------------------------------------------
    # Helper: compute per-sample sequence NLL and token counts (no grad).
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Logging helper for harmful vs benign statistics
    # ------------------------------------------------------------------
    def _log_harmful_stats(
        self,
        diff_scores: torch.Tensor,
        weights: torch.Tensor,
        harm_flags: torch.Tensor,
        compliance_seq_nll: torch.Tensor,
        refusal_seq_nll: torch.Tensor,
        *,
        view_label: str = "seq",  # "seq" for sequence-sum, "tok" for token-avg
    ) -> None:
        """Aggregate and per-sample logging split by harmful vs benign.

        Parameters
        ----------
        weights : Tensor  shape (B,)
        diff_scores : Tensor  shape (B,)
        harm_flags : Tensor  shape (B,)  with 1 = harmful, 0 = benign
        compliance_seq_nll : Tensor  (B,)
        refusal_seq_nll : Tensor  (B,)
        """
        # Respect system evaluation mode: do not log harmful stats when enabled
        if getattr(self.args, "system_evaluate", False):
            return

        try:
            harmful_idx = (harm_flags == 1).nonzero(as_tuple=False).flatten()
            benign_idx = (harm_flags == 0).nonzero(as_tuple=False).flatten()

            # Avg weights
            weight_harmful_avg = (
                weights[harmful_idx].mean().item() if harmful_idx.numel() > 0 else None
            )
            weight_benign_avg = (
                weights[benign_idx].mean().item() if benign_idx.numel() > 0 else None
            )

            # Avg diff scores
            score_harmful_avg = (
                diff_scores[harmful_idx].mean().item() if harmful_idx.numel() > 0 else None
            )
            score_benign_avg = (
                diff_scores[benign_idx].mean().item() if benign_idx.numel() > 0 else None
            )

            # Avg NLLs
            compliance_nll_harmful_avg = (
                compliance_seq_nll[harmful_idx].mean().item() if harmful_idx.numel() > 0 else None
            )
            compliance_nll_benign_avg = (
                compliance_seq_nll[benign_idx].mean().item() if benign_idx.numel() > 0 else None
            )
            refusal_nll_harmful_avg = (
                refusal_seq_nll[harmful_idx].mean().item() if harmful_idx.numel() > 0 else None
            )
            refusal_nll_benign_avg = (
                refusal_seq_nll[benign_idx].mean().item() if benign_idx.numel() > 0 else None
            )

            # Per-sample logging – one line per sample, sorted by descending weight
            sorted_idx = torch.argsort(weights, descending=True)
            for i in sorted_idx.tolist():
                comp_nll_val = compliance_seq_nll[i].item()
                ref_nll_val = refusal_seq_nll[i].item()
                logger.debug(
                    f"sample {i}: w={weights[i]:.4f}, d={diff_scores[i]:.4f}, comp_nll={comp_nll_val:.4f}, ref_nll={ref_nll_val:.4f}, h={int(harm_flags[i])}"
                )

            # Aggregate logs below
            logger.debug(
                f"Avg weight – harmful: {weight_harmful_avg}, benign: {weight_benign_avg}"
            )
            logger.debug(
                f"Avg diff_score (refusal-safe) – harmful: {score_harmful_avg}, benign: {score_benign_avg}"
            )
            logger.debug(
                f"Avg compliance_nll – harmful: {compliance_nll_harmful_avg}, benign: {compliance_nll_benign_avg}"
            )
            logger.debug(
                f"Avg refusal_nll – harmful: {refusal_nll_harmful_avg}, benign: {refusal_nll_benign_avg}"
            )

            # ------------------------------------------------------------
            # WandB logging for average weights (if active)
            # ------------------------------------------------------------
            if wandb.run is not None:
                wandb.log({
                    f"weight_harmful_avg_{view_label}": weight_harmful_avg,
                    f"weight_benign_avg_{view_label}": weight_benign_avg,
                })
        except Exception as e:
            logger.warning(f"Failed to compute / log harmful stats: {e}")

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
        if hasattr(self.args, "tau"):
            metrics_dict["tau"] = self.args.tau

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

    def compute_weighted_loss_and_gradient(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        weights: torch.Tensor,
        *,
        reduction: Optional[str] = "mean",  # kept for backward-compat; ignored internally
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Compute *token-level* weighted cross-entropy loss.

        This mirrors HF's default ``compute_loss`` (mean over **all** non-ignored tokens in
        the batch) while introducing a per-*sequence* weight vector.  Each token inside a
        sequence gets multiplied by its sequence's weight.  Formally::

            L = \frac{\sum_i w_i \sum_t CE(i,t)}{\sum_i w_i N_i}

        where ``CE(i,t)`` is the token cross-entropy for sample *i* token *t* and ``N_i`` is
        the number of non-ignored tokens in sample *i*.  When all ``w_i = 1`` this reduces
        exactly to the stock HF loss.
        """

        IGNORE_INDEX = -100

        # Forward pass
        with self.compute_loss_context_manager():
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            shift_logits = logits[..., :-1, :].contiguous()  # (B, L-1, V)

            labels = inputs.get("labels", inputs["input_ids"])  # fall back to input_ids
            shift_labels = labels[..., 1:].contiguous()  # (B, L-1)

            vocab_size = shift_logits.size(-1)
            ce_flat = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=IGNORE_INDEX,
            )  # (B*(L-1))

            ce = ce_flat.view(shift_labels.size())  # (B, L-1)
            token_mask = shift_labels.ne(IGNORE_INDEX).float()  # (B, L-1)

            if weights.size(0) != ce.size(0):
                raise ValueError("Weights size does not match batch size")

            # Broadcast sequence weights to token dimension and detach to avoid gradients.
            w = weights.detach().view(-1, 1).to(ce.dtype)  # (B, 1)

            weighted_token_losses = ce * token_mask * w        # numerator components
            weighted_token_mask = token_mask * w               # denominator components

            numerator = weighted_token_losses.sum()
            denominator = weighted_token_mask.sum().clamp(min=1.0)
            weighted_loss = numerator / denominator

        loss_value = weighted_loss.item()

        # Backward pass
        if self.use_apex:
            with amp.scale_loss(weighted_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(weighted_loss)

        # Collect gradients
        gradients = {
            name: param.grad.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

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

        # ------------------- Split paired batches (safe vs. refusal) -------------------
        # When the DataCollator adds refusal_* keys, we separate them here so that
        # (a) training continues to use the *safe* view for the actual forward/backward
        # (b) we can still log the refusal view for debugging.
        if "refusal_input_ids" in inputs:
            compliance_inputs = {k: v for k, v in inputs.items() if not k.startswith("refusal_")}

            # Strip the "refusal_" prefix so the dict matches the model’s expected signature
            refusal_inputs = {
                k[len("refusal_"):]: v for k, v in inputs.items() if k.startswith("refusal_")
            }
        else:
            compliance_inputs = inputs  # fallback – only compliance view in batch
            refusal_inputs = None

        # ------------------------------------------------------------------
        # Remove the harm-indicator before forwarding to the model. We keep a
        # copy in case future logic needs it, but the model forward should
        # never receive unknown kwargs.
        # ------------------------------------------------------------------
        harm_flags = compliance_inputs.get("is_harmful", None)
        if "is_harmful" in compliance_inputs:
            compliance_inputs = {k: v for k, v in compliance_inputs.items() if k != "is_harmful"}
        else:
            compliance_inputs = compliance_inputs

        logger.debug(
            f"Batch split -> compliance keys: {list(compliance_inputs.keys())} | refusal keys: {list(refusal_inputs.keys()) if refusal_inputs is not None else None}"
        )

        # ======================= Input Logging (Step 5 of Each Epoch) =======================
        # Skip heavy logging in system evaluation mode.
        if not getattr(self.args, "system_evaluate", False):
            # Advance counter with refusal view when present; otherwise log the safe view.
            self._maybe_log_inputs(compliance_inputs, dataset_type="compliance", advance_counter=True)
            if refusal_inputs is not None:
                self._maybe_log_inputs(refusal_inputs, dataset_type="refusal", advance_counter=False)

        # print(f"Refusal inputs: {refusal_inputs['input_ids'][0]}")

        # ---------------- Harmful-flag visualization ----------------
        if harm_flags is not None:
            try:
                harm_flags_cpu = harm_flags.detach().cpu()
                harmful_count = int((harm_flags_cpu == 1).sum().item())
                benign_count = int((harm_flags_cpu == 0).sum().item())
                logger.info(
                    f"[Batch {self.state.global_step}] is_harmful distribution – harmful: {harmful_count}, benign: {benign_count}, total: {harm_flags_cpu.numel()}"
                )
            except Exception as e:
                logger.warning(f"Failed to log harmful flag distribution: {e}")

        def step():
            # --- Weights & gradient computation ---------------------------
            # Always compute the *safe* view NLLs for downstream logging.
            compliance_seq_nll, safe_token_counts = self._sequence_nll(model, compliance_inputs)

            diff_scores = None  # initialise in case no refusal view exists
            # Use the temperature hyper-parameter from TrainingArguments if provided, else default to 1.0
            tau = getattr(self.args, "tau", 1.0)
            weights = None

            if refusal_inputs is not None:
                # Per-sample NLLs for the paired *refusal* view (used for weights & logging)
                refusal_seq_nll, refusal_token_counts = self._sequence_nll(model, refusal_inputs)

                # ---------------- Token-average diff & weights ----------------
                # Compute average CE per sample (avoid div/0)
                compliance_avg_ce = compliance_seq_nll / safe_token_counts.clamp(min=1)
                refusal_avg_ce = refusal_seq_nll / refusal_token_counts.clamp(min=1)

                # diff = avg_nll_refusal - avg_nll_compliance
                diff_scores = refusal_avg_ce - compliance_avg_ce  # (B,)

                # Softmax over the averaged diff scores to obtain weights
                weights = torch.softmax(diff_scores / tau, dim=0)
                
                # NEW: Set equal weights for benign samples, zero weights for harmful samples
                # Count benign samples (harm_flags == 0)
                # benign_mask = (harm_flags == 0)
                # num_benign = benign_mask.sum().item()
                
                # # Initialize weights tensor
                # weights = torch.zeros_like(harm_flags, dtype=torch.float32)
                
                # if num_benign > 0:
                #     # Set equal weights for benign samples
                #     weights[benign_mask] = 1.0 / num_benign
                    

                # uniform_w = torch.full_like(weights, 1.0 / weights.size(0))
                # weights = (weights + uniform_w) / 2.0  # smoothed weights
                # weights = uniform_w

                # Log harmful/benign statistics based on token-average values
                try:
                    self._log_harmful_stats(
                        diff_scores=diff_scores / tau,
                        weights=weights,
                        harm_flags=harm_flags,
                        compliance_seq_nll=compliance_avg_ce,
                        refusal_seq_nll=refusal_avg_ce,
                        view_label="tok",
                    )
                except Exception as e:
                    logger.warning(f"Failed to compute token-avg weights/diff: {e}")


            # Compute weighted gradient if weights exist and match batch size
            if refusal_inputs is not None:
                weighted_loss_value, weighted_grads = self.compute_weighted_loss_and_gradient(
                    model,
                    compliance_inputs,
                    weights,
                    reduction="mean"
                )

                # Replace param.grad with weighted grads
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.grad.data = weighted_grads[name]

                final_loss_scalar = weighted_loss_value
            else:
                # Fallback to unweighted loss (original behaviour)
                alignment_safe_loss, safe_grads = self.compute_loss_and_gradient(
                    model, compliance_inputs, zero_grad=False
                )

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.grad.data = safe_grads[name]

                final_loss_scalar = alignment_safe_loss


            # ---------------------- Weight computation ----------------------
            # The weights are now computed and applied within the step function.
            # The original code had a redundant call here.

            self.steps += 1

            # Compute and log metrics only every 10 steps (skip in system evaluation mode)
            if (self.steps % 10 == 0) and (not getattr(self.args, "system_evaluate", False)):
                # ------------------ Normal (unweighted) loss ------------------
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        normal_loss_tensor = self.compute_loss(model, compliance_inputs)
                    normal_loss_scalar = normal_loss_tensor.item()

                final_grads = {
                    name: param.grad.data.clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                final_grad_norm = self._grad_norm(final_grads) + 1e-7
                metrics_dict = {
                    "step": self.steps,
                    "final_grad_norm": final_grad_norm.item(),
                    "normal_loss_ce": normal_loss_scalar,
                    "alignment_safe_loss": final_loss_scalar,
                    "compliance_avg_ce": compliance_avg_ce.mean().item(),
                    "refusal_avg_ce": refusal_avg_ce.mean().item(),
                }

                # Log all metrics
                self._log_metrics(metrics_dict)

            final_loss = final_loss_scalar
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

            # Ensure "input_ids" key is present
            if "input_ids" not in inputs:
                raise ValueError("input_ids not found in inputs")

            # -----------------------------
            # Iterate over *all* samples in the batch so we can inspect each
            # instruction / response pair individually (useful for refusal view).
            # -----------------------------
            batch_size = inputs["input_ids"].size(0)

            for sample_idx in range(batch_size):
                try:
                    # Slice per-sample tensors to create a lightweight view that
                    # is consistent across keys (labels, masks, etc.). For non-tensor
                    # entries we simply pass the original value.
                    single_sample_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            # Only slice along batch dimension when it matches.
                            if v.dim() > 0 and v.size(0) == batch_size:
                                # Keep the leading batch dimension so downstream logging utils
                                # (which expect inputs[foo][0]) still work.
                                single_sample_inputs[k] = v[sample_idx : sample_idx + 1]
                            else:
                                single_sample_inputs[k] = v
                        else:
                            single_sample_inputs[k] = v

                    # Use a 1-D tensor of token IDs for decoding (tokenizer.decode expects list[int] / 1-D tensor)
                    sample_input_ids = single_sample_inputs["input_ids"][0]

                    # Log harm flag if present for this sample
                    if "is_harmful" in single_sample_inputs:
                        try:
                            harm_flag = single_sample_inputs["is_harmful"].item()
                            logger.info(f"is_harmful flag (sample {sample_idx}): {harm_flag}")
                        except Exception as e:
                            logger.warning(f"Failed to log is_harmful flag for sample {sample_idx}: {e}")

                    # Ensure that we have both a sample and a tokenizer.
                    if sample_input_ids is None or getattr(self, "tokenizer", None) is None:
                        continue

                    # Perform the actual logging via utility for this sample.
                    log_input_data_details(
                        sample_input_ids=sample_input_ids,
                        inputs=single_sample_inputs,
                        tokenizer=self.tokenizer,
                        current_epoch=current_epoch_idx,
                        step_in_epoch=self._log_step_in_epoch,
                        global_step=self.state.global_step,
                        dataset_type=f"{dataset_type}_sample_{sample_idx}",
                    )
                except Exception as e:
                    logger.warning(
                        f"Input logging failed for sample {sample_idx} at epoch {self.state.epoch}, global step {self.state.global_step}: {e}"
                    )

        except Exception as e:
            # Protect training from any logging failures.
            logger.warning(
                f"Input logging failed at epoch {self.state.epoch}, global step {self.state.global_step}: {e}"
            )