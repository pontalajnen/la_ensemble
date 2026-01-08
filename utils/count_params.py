import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.resnet20_frn_packed import ResNet20_FRN_packed
from models.resnet20_frn import ResNet20_FRN


def count_parameters(state_dict):
    """Count total parameters from state dict."""
    # parameter_count = sum(p.numel() for p in state_dict.values())
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {parameter_count:,}")
    print(f"Total (M): {parameter_count / 1e6:.2f}M")

    return parameter_count


model = ResNet20_FRN_packed(num_classes=10)
model = ResNet20_FRN(num_classes=10)
# model = torch.load(sys.argv[1], map_location='cpu')
total = count_parameters(model)
