import torch
import torch.nn as nn
import argparse
from raft_hidden import RAFT

# Assuming `model` is your PyTorch model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


parser = argparse.ArgumentParser()
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
parser.add_argument('--use_enhance', action='store_true', default=True)
parser.add_argument('--result_path', default='runs/debug')
parser.add_argument('--extractor_denoise', action='store_true', default=False)
parser.add_argument('--dataset_root', default='/home/whx/code/Retinexformer/data/VBOF/VBOF_dataset')
args = parser.parse_args()


model = torch.nn.DataParallel(RAFT(args))
param_count = count_parameters(model)

# Display the number of parameters in millions (M)
print(f"Total Parameters: {param_count:,} ({param_count / 1e6:.2f} M)")
