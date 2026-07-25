import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(input_size, 128), 
                                     nn.ReLU(), 

                                     nn.Linear(128, 128), 
                                     nn.ReLU(), 

                                     nn.Linear(128, output_size))

    def forward(self, x):
        return self.network(x)
    
