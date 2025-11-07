import os
import torch


def print_info(script, text):
    script = script.split("/")[-1]
    padding = (max([len(file) for file in os.listdir(".")]) - len(script)) * " "
    print(f"[{script}] {padding} ----- {text} -----")


def torch_device():
    return torch.device(
        'cuda:0' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
