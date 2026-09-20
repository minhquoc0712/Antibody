from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import time
import torch
import collections
import copy
import random
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn
from loss_func.repnoise_loss import rep_noise_loss
from transformers import Trainer
from transformers import logging

# from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import is_sagemaker_mp_enabled

# from utils import prune_wanda_outlier,SupervisedDataset,prune_with_FI
import wandb

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, RandomSampler

# from transformers.models.falcon.modeling_falcon import FalconAttention
# from transformers.models.mistral.modeling_mistral import MistralAttention

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

# Import apex amp for mixed precision training
try:
    from apex import amp
except ImportError:
    amp = None

# if is_torch_tpu_available():
#     import torch_xla.core.xla_model as xm
#     import torch_xla.debug.metrics as met
#     import torch_xla.distributed.parallel_loader as pl


from loguru import logger


class LisaTrainer(Trainer):

    def get_alignment_dataloader(self, alignment_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        from transformers.trainer_utils import seed_worker
        from transformers.trainer_pt_utils import (
            LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler

        data_collator = self.data_collator

        sampler = RandomSampler(alignment_dataset)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(
            DataLoader(alignment_dataset, **dataloader_params)
        )

    def init(self, alignment_dataset):
        if self.args.alignment_step != 0 and self.args.guide_data_num > 0:
            self.status = "alignment"
        else:
            self.status = "finetune"
        
        # Log hyperparameters to wandb
        if wandb.run is not None:
            wandb.config.update({
                "rho": getattr(self.args, 'rho', 0),
                "guide_data_num": getattr(self.args, 'guide_data_num', 0),
                "alignment_step": getattr(self.args, 'alignment_step', 0),
                "finetune_step": getattr(self.args, 'finetune_step', 0),
            })
            logger.info("Hyperparameters logged to wandb")
        else:
            logger.warning("wandb not initialized - skipping hyperparameter logging")
            logger.info(f"LISA Trainer initialized with rho={getattr(self.args, 'rho', 0)}, guide_data_num={getattr(self.args, 'guide_data_num', 0)}, alignment_step={getattr(self.args, 'alignment_step', 0)}, finetune_step={getattr(self.args, 'finetune_step', 0)}")
            logger.info(f"Starting in {self.status} mode")
        
        self.alignment_weights = {}
        self.finetune_weights = {}
        # Flags to track first-time proximal term usage
        self.alignment_proximal_started = False
        self.finetune_proximal_started = False
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.alignment_weights[name] = param.data.detach().clone()
                self.finetune_weights[name] = param.data.detach().clone()
                # self.gamma[name]= torch.zeros_like(param)
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num > 0:
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_dataset)
            self.data_iter = iter(self.alignment_dataloader)

    def end_training(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.status == "alignment":
                    self.alignment_weights[name] = param.data.detach().clone()
                else:
                    self.finetune_weights[name] = param.data.detach().clone()

    def switch_model(self):
        sum_drift = 0
        if self.status == "alignment":
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.finetune_weights[name] = param.data.detach().clone()
                    sum_drift += (
                        torch.norm(
                            self.finetune_weights[name] - self.alignment_weights[name]
                        )
                        ** 2
                    )
            print("finetuning drift to consensus{}".format(sum_drift))
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.alignment_weights[name] = param.data.detach().clone()
                    sum_drift += (
                        torch.norm(
                            self.finetune_weights[name] - self.alignment_weights[name]
                        )
                        ** 2
                    )
            print("alignment drift to consensus{}".format(sum_drift))

    def sample_from_alignment(self):
        # Get a  batch
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # If the iterator is exhausted, create a new iterator
            self.data_iter = iter(self.alignment_dataloader)
            batch = next(self.data_iter)
        return batch

    def check_mode(self, inputs):
        if self.status == "alignment":
            if (
                self.clock % (self.args.alignment_step) == 0
                and self.steps != 0
                and self.args.finetune_step != 0
            ):
                self.status = "finetune"
                self.switch_model()
                logger.info(f"Phase change: switched from alignment to finetune at step {self.steps}")
                # print("swith from alignment to finetune {}".format(self.steps))
                self.clock = 0

            else:
                # alignment need another input
                inputs = self.sample_from_alignment()
        else:
            if (
                self.clock % (self.args.finetune_step) == 0
                and self.steps != 0
                and self.args.alignment_step != 0
                and self.args.guide_data_num > 0
            ):
                self.status = "alignment"
                self.switch_model()
                logger.info(f"Phase change: switched from finetune to alignment at step {self.steps}")
                # alignment need another input

                inputs = self.sample_from_alignment()
                # print("swith from finetune to alignment {}".format(self.steps))
                self.clock = 0
        return inputs

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        inputs = self.check_mode(inputs)
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
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.status == "alignment":
                # print("alignment_loss_prev: {}".format(loss.item()))
                if (
                    self.steps
                    > 0.1
                    * len(self.get_train_dataloader())
                    * self.args.num_train_epochs
                ):
                    if not self.alignment_proximal_started:
                        logger.info(f"Proximal term: started using proximal regularization in alignment mode at step {self.steps}")
                        self.alignment_proximal_started = True
                    
                    for name, param in model.named_parameters():
                        if param.requires_grad and self.args.rho > 0:
                            # loss +=torch.sum(self.gamma[name] *  param)+ self.args.rho/2* torch.norm( param- self.finetune_weights[name])**2
                            loss += (
                                self.args.rho
                                / 2
                                * torch.norm(param - self.finetune_weights[name]) ** 2
                            )
                # print("alignment_loss: {}".format(loss.item()))
            else:
                # print("finetune_loss_prev: {}".format(loss.item()))

                if (
                    self.steps
                    > 0.1
                    * len(self.get_train_dataloader())
                    * self.args.num_train_epochs
                ):
                    if not self.finetune_proximal_started:
                        logger.info(f"Proximal term: started using proximal regularization in finetune mode at step {self.steps}")
                        self.finetune_proximal_started = True
                    
                    for name, param in model.named_parameters():
                        # we observe that for Gsm8k, proximal term will hurt convergence. Don't do proximal for the first few rounds.
                        if param.requires_grad and self.args.rho > 0:
                            # loss += (- torch.sum(self.gamma[name] *  param )) + self.args.rho/2* torch.norm( param- self.alignment_weights[name])**2
                            loss += (
                                self.args.rho
                                / 2
                                * torch.norm(param - self.alignment_weights[name]) ** 2
                            )
                # print("finetune_loss: {}".format(loss.item()))
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss

        loss = step()
        self.steps += 1
        self.clock += 1
        return loss.detach() / self.args.gradient_accumulation_steps
