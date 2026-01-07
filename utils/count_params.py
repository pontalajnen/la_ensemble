import torch
import sys


def count_parameters(state_dict):
    """Count total parameters from state dict."""
    parameter_count = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {parameter_count:,}")
    print(f"Total (M): {parameter_count / 1e6:.2f}M")

    return parameter_count


model = torch.load(sys.argv[1], map_location='cpu')
total = count_parameters(model)
