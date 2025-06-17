import torch.nn as nn
import random
import torch
from resnet import torch_resnet18


class EnsembleModel(nn.Module):
    def __init__(self, num_models=5, num_classes=10, model=torch_resnet18):
        super().__init__()
        seeds = [x for x in range(num_models)]

        self.models = nn.ModuleList()
        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            random.seed(seed)

            m = model(num_classes)
            self.models.append(m)

    def forward(self, x: torch.Tensor):
        return [m(x) for m in self.models]
