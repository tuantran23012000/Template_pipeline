import torch
from torchvision import datasets, transforms
import wandb
import argparse
from dataset.transform import get_transform
from models.model import Net
from models.loss import NllLoss
from models.optimizer import Optimizer
from models.scheduler import Scheduler
from configs import cfg, update_config
import yaml
class Train_pipeline():
    def __init__(self,args,cfgs,mode):
        self.cfgs = cfgs
        self.args = args
        self.mode = mode
        # setting config to log on wandb
        if self.mode == "train":
            config = {
                "epochs": self.cfgs["TRAIN"]["EPOCH"],
                "learning_rate_init": self.cfgs["TRAIN"]["OPTIMIZER"]["LR"],
                "batch_size": self.cfgs["TRAIN"]["BATCH_SIZE"]
            }
            wandb.init(project = self.cfgs["PROJECT_NAME"], entity = "ai-iot",config=config)
        # setting device 
        self.gpu = self.cfgs["TRAIN"]["GPUS"]["USE_GPU"]
        if self.gpu:
            self.gpu_id = self.cfgs["TRAIN"]["GPUS"]["GPU_ID"]
            self.device = torch.device("cuda:"+self.gpu_id)
    def load_data(self):
        # load data and preprocessing
        train_transform, test_transform = get_transform(self.cfgs)
        train_mnist = datasets.MNIST('../data',train=True,download = True,transform = train_transform)
        test_mnist = datasets.MNIST('../data',train=False,transform = test_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset = train_mnist,
            batch_size = self.cfgs["TRAIN"]["BATCH_SIZE"],
            shuffle = True,
            num_workers = self.cfgs["TRAIN"]["NUM_WORKERS"]
            )
        test_loader = torch.utils.data.DataLoader(
            dataset = test_mnist,
            batch_size = self.cfgs["TRAIN"]["BATCH_SIZE"],
            shuffle = False,
            num_workers = self.cfgs["TRAIN"]["NUM_WORKERS"]
            )
        return train_loader, test_loader
    def make_model(self):
        # create model
        model = Net()
        model = model.to(self.device)
        return model
    def make_loss(self):
        # create loss function
        loss = NllLoss(self.cfgs)
        return loss
    def make_optimizer(self,model):
        # create optimizer
        optimizer = Optimizer(self.cfgs, model).get_optim()
        return optimizer
    def make_scheduler(self, optimizer):
        # create scheduler
        scheduler = Scheduler(self.cfgs, optimizer).get_scheduler()
        return scheduler
    def train_step(self, model, optimizer):
        # train per batch
        model.train()
        train_loader, _ = self.load_data()
        optimizer = self.make_optimizer(model)
        loss = self.make_loss()
        loss_per_batch = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss_batch = loss(output, target)
            loss_batch.backward()
            optimizer.step()
            loss_per_batch += loss_batch.item()
        loss_per_batch /= len(train_loader.dataset)
        print("train loss:", loss_per_batch)
        return loss_per_batch
    def valid_step(self, model):
        # eval per batch
        model.eval()
        _, test_loader = self.load_data()
        test_acc = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim = 1, keepdim = True)
                test_acc += pred.eq(target.view_as(pred)).sum().item()
        test_acc /= len(test_loader.dataset)
        print("test acc:", test_acc)
        return test_acc
    def epoch_step(self):
        epoch = self.cfgs["TRAIN"]["EPOCH"]
        model = self.make_model()
        optimizer = self.make_optimizer(model)
        scheduler = self.make_scheduler(optimizer)
        for e in range(1, epoch + 1):
            print("Epoch: ",e)
            train_loss = self.train_step(model, optimizer)
            test_acc = self.valid_step(model)
            scheduler.step()
            log_dict = {
                "Train loss": train_loss,
                "Test accuracy": test_acc
            }
            wandb.log(log_dict)
            save_dict = {
                'state_dicts': model.state_dict()
            }
        torch.save(save_dict, self.cfgs["SAVE_MODEL_DIR"])
        art = wandb.Artifact(f'mnist-{wandb.run.id}',type = "model")
        art.add_file(self.cfgs["SAVE_MODEL_DIR"], "last_model.pkl")
        wandb.log_artifact(art)
def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/mnist.yaml",
    )
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    # update_config(cfg, args)
    with open(args.cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)
    print(cfg)
    bot = Train_pipeline(args,cfg,mode = args.mode)
    if args.mode == "train":
        bot.epoch_step()

        
