import torch.optim as optim
class Optimizer():
    def __init__(self,cfgs, model):
        self.cfgs = cfgs
        self.model = model
        self.param_groups = [{
            'params':self.model.parameters(),
            'lr': self.cfgs["TRAIN"]["OPTIMIZER"]["LR"],
            #'weight_decay': self.cfgs["TRAIN"]["OPTIMIZER"]["WEIGHT_DECAY"]
        }]
    def get_optim(self):
        if self.cfgs["TRAIN"]["OPTIMIZER"]["TYPE"].lower() == "sgd":
            optimizer = optim.SGD(self.param_groups, momentum = self.cfgs["TRAIN"]["OPTIMIZER"]["MOMENTUM"])
        elif self.cfgs["TRAIN"]["OPTIMIZER"]["TYPE"].lower() == "adam":
            optimizer = optim.Adam(self.param_groups)
        elif self.cfgs["TRAIN"]["OPTIMIZER"]["TYPE"].lower() == "adamw":
            optimizer = optim.AdamW(self.param_groups)
        elif self.cfgs["TRAIN"]["OPTIMIZER"]["TYPE"].lower() == "adadelta":
            optimizer = optim.Adadelta(self.param_groups)
        return optimizer