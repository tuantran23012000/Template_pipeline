import torch
import torchvision.transforms as T
def get_transform(cfg):
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,),(0.3081,))
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,),(0.3081,))
    ])
    return train_transform, test_transform