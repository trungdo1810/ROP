# Custom GradualWarmupSchedulerV2
class GradualWarmupSchedulerV2:
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.optimizer = optimizer
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        if self.last_epoch <= self.total_epoch:
            # Warmup phase
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        elif self.after_scheduler:
            # Transition to after_scheduler
            if not self.finished:
                self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                self.finished = True
            self.after_scheduler.step(self.last_epoch - self.total_epoch)
    
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]