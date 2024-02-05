import torch
import math
import timm
import torchvision

class AE_BEST(torch.nn.Module):
    """AE_Default with pretrained weights based on an autoencoding task."""
    def __init__(self):
        super().__init__()
        self.net = torch.load("/home/jdolli/sentinel2_foundation_model/best_model.pt").encoder.to("cpu")
        
    def forward(self, x):
        return self.net(x)

    
class AE_DEFAULT(torch.nn.Module):
    """Default for Sat-SINR. Used as autoencoder in a previous project inspired by beta-VAE"""
    def __init__(self, hidden_dim=128):
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
            torch.nn.Linear(256, hidden_dim*2),
        ).to("cpu")
        
    def forward(self, x):
        return self.net(x)
    
class CNN_SMALLERINPUT(torch.nn.Module):
    """AE_Default, but reducing the receptive field"""
    def __init__(self, layer_removed=1, hidden_dim=128):
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
            torch.nn.Linear(256, hidden_dim*2)
        )
        
    def forward(self, x):
        return self.net(self.center_crop(x, 128 // math.pow(2, self.layer_removed)))

class Resnet(torch.nn.Module):
    def __init__(self, v="34"):
        super().__init__()
        self.net = timm.create_model('resnet' + v, num_classes=256)
        self.net.conv1 = torch.nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(1, 4, 128, 128)
        return self.net(x)
    
class VGG(torch.nn.Module):
    def __init__(self, v="11"):
        super().__init__()
        self.net = timm.create_model('vgg' + v, num_classes=256)
        self.net.features[0] = torch.nn.Conv2d(4, 64, 3, 1, 1)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(1, 4, 128, 128)
        return self.net(x)

    
class View(torch.nn.Module):
    # Taken from https://github.com/1Kotorch.nny/Beta-VAE/blob/master/model.py
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

    
class VIT_SMALL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = timm.create_model('vit_small_patch8_224', num_classes=256)
        self.net.patch_embed.proj = torch.nn.Conv2d(4, 384, 8, 8)
        self.crop = torchvision.transforms.functional.center_crop
        
    def forward(self, x):
        x = self.crop(x, 224)
        if len(x.shape) == 3:
            x = x.view(1, 4, 224, 224)
        return self.net(x)
    
class VIT_SMALL_DINO_MC(VIT_SMALL):
    def __init__(self):
        super().__init__()
        state_dict = torch.load("vit_mc_checkpoint300.pth")["student"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        self.net.patch_embed.proj = torch.nn.Conv2d(3, 384, 8, 8)
        self.net.load_state_dict(state_dict, strict=False)
        self.net.patch_embed.proj = torch.nn.Conv2d(4, 384, 8, 8)
        
    def forward(self, x):
        x = self.crop(x, 224)
        if len(x.shape) == 3:
            x = x.view(1, 4, 224, 224)
        return self.net(x)
    
def get_embedder(params):
    if params.embedder == "ae_best":
        return AE_BEST()
    elif params.embedder == "ae_default":
        return AE_DEFAULT()
    elif params.embedder == "resnet18":
        return Resnet("18")
    elif params.embedder == "resnet34":
        return Resnet("34")
    elif params.embedder == "resnet50":
        return Resnet("50")
    elif params.embedder == "vgg11":
        return VGG("11")
    elif params.embedder == "vgg13":
        return VGG("13")
    elif params.embedder == "vgg19":
        return VGG("19")
    elif params.embedder == "vit_small":
        return VIT_SMALL()
    elif params.embedder == "dino_mc_vm":
        return VIT_SMALL_DINO_MC()
    elif params.embedder.startswith("cnn_si"):
        return CNN_SMALLERINPUT(int(params.embedder[-1]))
    else:
        raise NotImplementedError