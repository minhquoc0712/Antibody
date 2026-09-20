from transformers import TrainerCallback
import torch


class GPUTimeCallback(TrainerCallback):
    def __init__(self, total_steps):
        super().__init__()
        self.average_statistic = 0
        self.record_time = 0
        self.total_steps = total_steps

    def on_step_begin(self, args, state, control, **kwargs):
        state.start_event = torch.cuda.Event(enable_timing=True)
        state.end_event = torch.cuda.Event(enable_timing=True)
        state.start_event.record()

    def on_step_end(self, args, state, control, **kwargs):
        state.end_event.record()
        torch.cuda.synchronize()
        step_time = state.start_event.elapsed_time(state.end_event)
        self.average_statistic = (
            self.average_statistic * self.record_time + step_time
        ) / (self.record_time + 1)
        self.record_time += 1
        if self.record_time % 100 == 0:
            # print(f"Step {state.global_step}: {self.average_statistic*self.record_time / 1000:.2f} seconds (GPU time)")
            print(
                "Estimated total time {} (h)".format(
                    self.average_statistic * self.total_steps / 1000 / 3600
                )
            )


class GPUMemoryCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.average_statistic_memory = 0
        self.record_time_memory = 0

    def on_step_begin(self, args, state, control, **kwargs):
        state.start_memory = torch.cuda.memory_reserved()
        # print(self.record_time_memory)

    def on_step_end(self, args, state, control, **kwargs):
        state.end_memory = torch.cuda.memory_reserved()
        self.average_statistic_memory = (
            self.average_statistic_memory * self.record_time_memory + state.end_memory
        ) / (self.record_time_memory + 1)
        self.record_time_memory += 1
        if self.record_time_memory % 100 == 0:
            print(
                f"Step {state.global_step}: {self.average_statistic_memory / (1024 ** 3):.2f} GB GPU memory used"
            )


class GPUPeakMemoryCallback(TrainerCallback):
    def __init__(self, reset_each_step=True, log_every_n_steps=100):
        super().__init__()
        self.reset_each_step = reset_each_step
        self.log_every_n_steps = log_every_n_steps
        self.training_peak_reserved = 0
        self.training_peak_allocated = 0
        self.global_step_counter = 0

    def _device(self):
        if not torch.cuda.is_available():
            return None
        return torch.cuda.current_device()

    def _reset_peaks(self):
        device = self._device()
        if device is None:
            return
        torch.cuda.reset_peak_memory_stats(device=device)

    def on_train_begin(self, args, state, control, **kwargs):
        # Reset training-wide peaks at the beginning
        if not torch.cuda.is_available():
            return
        self.training_peak_reserved = 0
        self.training_peak_allocated = 0
        self._reset_peaks()

    def on_step_begin(self, args, state, control, **kwargs):
        # Reset per-step peak stats if requested
        if self.reset_each_step:
            self._reset_peaks()

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        self.global_step_counter += 1

        device = self._device()
        peak_reserved = torch.cuda.max_memory_reserved(device=device)
        peak_allocated = torch.cuda.max_memory_allocated(device=device)

        # Update training-wide peaks
        if peak_reserved > self.training_peak_reserved:
            self.training_peak_reserved = peak_reserved
        if peak_allocated > self.training_peak_allocated:
            self.training_peak_allocated = peak_allocated

        if self.global_step_counter % self.log_every_n_steps == 0:
            print(
                f"Step {state.global_step}: peak_reserved={peak_reserved / (1024 ** 3):.2f} GB, "
                f"peak_allocated={peak_allocated / (1024 ** 3):.2f} GB"
                f"Global Peak Reserved={self.training_peak_reserved / (1024 ** 3):.2f} GB, "
                f"Global Peak Allocated={self.training_peak_allocated / (1024 ** 3):.2f} GB"
            )

    def on_train_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        print("Training-wide GPU Peak Memory Usage (single GPU):")
        print(
            f" - peak_reserved={self.training_peak_reserved / (1024 ** 3):.2f} GB, "
            f"peak_allocated={self.training_peak_allocated / (1024 ** 3):.2f} GB"
        )
