import torch.nn as nn
import random
import torch
from .resnet import ResNet18


class EnsembleModel(nn.Module):
    def __init__(self, model=ResNet18, num_models=4, num_classes=10):
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
        predictions = [m(x) for m in self.models]

        if self.training:
            return predictions
        return torch.stack(predictions, dim=0).mean(dim=0)
