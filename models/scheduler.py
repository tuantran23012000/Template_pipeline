from torch.optim.lr_scheduler import StepLR

class Scheduler():
    def __init__(self, cfgs, optimizer):
        self.cfgs = cfgs
        self.optimizer = optimizer
    def get_scheduler(self):
        if self.cfgs["TRAIN"]["LR_SCHEDULER"]["TYPE"] == "steplr":
            scheduler = StepLR(self.optimizer, step_size = 1,gamma = self.cfgs["TRAIN"]["LR_SCHEDULER"]["GAMMA"])
        return scheduler