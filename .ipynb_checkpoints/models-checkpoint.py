import torch

class ResidLayer(torch.nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        b = self.layers(x)
        x = x + b
        return x

class SINR_Net(torch.nn.Module):
    def __init__(self, input_len=4, hidden_dim = 256, dropout = 0.5, layers = 4):
        super().__init__()
        
        self.location_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_len, hidden_dim),
            torch.nn.ReLU(),
            *[ResidLayer(hidden_dim, dropout) for i in range(layers)]
        )
        
        self.classifier = self.net = torch.nn.Linear(hidden_dim, 10040)
        
    def forward(self, x):
        x = self.location_encoder(x)
        x = self.classifier(x)
        return x
