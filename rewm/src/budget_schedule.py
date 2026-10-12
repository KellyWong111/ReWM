import math
from torch.optim.lr_scheduler import LRScheduler

class BudgetCosine(LRScheduler):
    """Warm up, hold the peak, cosine decay to a floor, then hold the floor."""
    def __init__(self, optimizer, warmup_steps, hold_until, decay_end, min_lr_ratio, last_epoch=-1):
        assert 0 <= warmup_steps <= hold_until < decay_end
        assert 0 < min_lr_ratio < 1
        self.warmup_steps = warmup_steps
        self.hold_until = hold_until
        self.decay_end = decay_end
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            ratio = step / self.warmup_steps
        else:
            progress = min(1.0, max(0.0, (step-self.hold_until)/(self.decay_end-self.hold_until)))
            ratio = self.min_lr_ratio + (1-self.min_lr_ratio)*(1+math.cos(math.pi*progress))/2
        return [base * ratio for base in self.base_lrs]
