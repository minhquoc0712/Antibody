from typing import Any, Dict, Union
import torch
import torch.nn as nn
from packaging import version
from transformers import Trainer
from transformers import logging
from transformers.utils import is_sagemaker_mp_enabled

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers.models.gemma.modeling_gemma import GemmaAttention
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention

try:
    from apex import amp
except ImportError:
    amp = None

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast  # noqa: F401

from loguru import logger
import wandb


def get_leaf_modules_with_grad(module):
    module_list = []
    for name, module in module.named_modules():
        if (
            isinstance(module, LlamaAttention)
            or isinstance(module, OPTAttention)
            or isinstance(module, MistralAttention)
            or isinstance(module, GemmaAttention)
            or isinstance(module, Qwen2Attention)
            or isinstance(module, Gemma2Attention)
        ):
            module_list += [module]
    return module_list


class VaccineTrainer(Trainer):
    def init(self):
        self.clock = 0
        self.steps = 0
        # Initialise wandb run if not already started
        if wandb.run is None:
            project_name = "vaccine"
            run_name = f"vaccine_rho{getattr(self.args, 'rho', None)}"
            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "rho": getattr(self.args, "rho", None),
                },
            )

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(
                    model, inputs, self.args.gradient_accumulation_steps
                )
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        # Two-step Vaccine-style update using activation perturbations
        self.sam_state = {}
        self.sam_state["hooks"] = []
        self.sam_state["gradient"] = {}
        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)

        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model: nn.Module):
        def track_gradient_hook(module, grad_input, grad_output):
            self.sam_state["gradient"][module] = (
                grad_output[0].detach().clone() / self.args.gradient_accumulation_steps
            )

        def apply_backward_hooks(mod: nn.Module, hook_fn, hooks):
            hook = mod.register_backward_hook(hook_fn)
            hooks.append(hook)

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def pre_second_step(self, model: nn.Module):
        def perturbation_hook(module, input, output):
            perturbation = self.sam_state["gradient"][module]
            output[0].data = output[0] + perturbation
            return output

        def apply_forward_hooks(mod: nn.Module, hook_fn, hooks):
            hook = mod.register_forward_hook(hook_fn)
            hooks.append(hook)

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            apply_forward_hooks(layer, perturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model: nn.Module):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

        grad_norm = self._grad_norm(self.sam_state["gradient"])
        for module in self.sam_state["gradient"]:
            grad = self.sam_state["gradient"][module]
            scale = self.args.rho / (grad_norm + 1e-7)
            e_r = grad * scale
            self.sam_state["gradient"][module] = e_r.detach().clone()

    @torch.no_grad()
    def after_second_step(self, model: nn.Module):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

    @torch.no_grad()
    def _grad_norm(
        self, grads_by_module: Dict[nn.Module, torch.Tensor]
    ) -> torch.Tensor:
        return torch.norm(
            torch.stack(
                [(grads_by_module[module]).norm(p=2) for module in grads_by_module]
            ),
            p=2,
        )

    def _log_metrics(self, metrics_dict: Dict[str, Any]):
        """Log metrics to console and wandb, similar to BoosterTrainer.

        Args:
                metrics_dict: Dictionary of metrics to log
        """
        # Work on a shallow copy to avoid mutating caller data
        metrics = dict(metrics_dict)

        # Attach common hyperparameters if present
        if hasattr(self.args, "rho"):
            metrics["rho"] = self.args.rho

        # Determine step for logging
        step = getattr(self.state, "global_step", None)
        if step is not None:
            metrics["step"] = step

        # Console print
        print(f"\n===== Step {step} Metrics =====", flush=True)
        for key, value in metrics.items():
            if key != "step":
                print(f"{key}: {value}", flush=True)

        # Wandb log if available
        if wandb.run is not None:
            wandb.log(metrics)
