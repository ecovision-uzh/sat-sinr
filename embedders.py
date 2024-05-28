import torch
import math
import torchvision

    
class CNN_DEFAULT(torch.nn.Module):
    """Default for Sat-SINR. Used as autoencoder in a previous project inspired by beta-VAE"""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 4, 2, 1),
            torch.nn.ReLU(),
            View((-1, 512)),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
        )
        
    def forward(self, x):
        return self.net(x)
    
class CNN_SMALLERINPUT(torch.nn.Module):
    """AE_Default, but reducing the receptive field"""
    def __init__(self, layer_removed=1):
        super().__init__()
        self.center_crop = torchvision.transforms.functional.center_crop
        self.layer_removed = layer_removed
        layers = [torch.nn.Conv2d(4, 32, 4, 2, 1),
            torch.nn.ReLU()]
        for i in range(layer_removed):
            layers.append(torch.nn.Conv2d(32, 32, 3, 1, 1))
            layers.append(torch.nn.ReLU())
        for i in range(4 - layer_removed):
            layers.append(torch.nn.Conv2d(32, 32, 4, 2, 1))
            layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(
            *layers,
            View((-1, 512)),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256)
        )
        
    def forward(self, x):
        return self.net(self.center_crop(x, 128 // math.pow(2, self.layer_removed)))


class View(torch.nn.Module):
    # Taken from https://github.com/1Kotorch.nny/Beta-VAE/blob/master/model.py
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

    
def get_embedder(params):
    if params.embedder == "cnn_default":
        return CNN_DEFAULT()
    elif params.embedder.startswith("cnn_si"):
        return CNN_SMALLERINPUT(int(params.embedder[-1]))
    else:
        raise NotImplementedError