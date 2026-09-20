from typing import Any, Dict, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer
from transformers.trainer_utils import seed_worker
from transformers.utils import is_sagemaker_mp_enabled

from loss_func.repnoise_loss import rep_noise_loss

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

try:
    from apex import amp
except ImportError:
    amp = None


class RepNoiseTrainer(Trainer):
    def init(self, harmful_dataset):
        data_collator = self.data_collator
        sampler = RandomSampler(harmful_dataset)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(harmful_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        self.harmful_dataloader = self.accelerator.prepare(
            DataLoader(harmful_dataset, **dataloader_params)
        )

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        data_iter = iter(self.harmful_dataloader)
        harmful_inputs = next(data_iter)
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(
                    model, inputs, self.args.gradient_accumulation_steps
                )
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = rep_noise_loss(
                    model,
                    harmful_inputs,
                    inputs,
                    beta=self.args.lamb,
                    alpha=self.args.rho,
                )
            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps
