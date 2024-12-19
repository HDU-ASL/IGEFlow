import torch
import torch.nn as nn
import argparse
# from raft_hidden import RAFT
from raft import RAFT
import os
import datasets
from tqdm import tqdm
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


# 注册钩子函数，保存特征图
features = {}

def hook(module, input, output):
    features['hooked_layer'] = output

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint", default='runs/origin/checkpoints/origin.pth')
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
parser.add_argument('--use_enhance', action='store_true', default=True)
parser.add_argument('--result_path', default='runs/debug/PIC/raft')
parser.add_argument('--extractor_denoise', action='store_true', help='use extractor_denoise model')
parser.add_argument('--dataset_root', default='/home/whx/code/Retinexformer/data/VBOF/VBOF_dataset')
args = parser.parse_args()

# 初始化并加载模型
model = torch.nn.DataParallel(RAFT(args))
state_dict = torch.load(args.model)
model.load_state_dict(state_dict)

model.cuda()
model.eval()

# 为某一层注册hook（假设我们要可视化 `module.some_layer`）
model.module.fnet.register_forward_hook(hook)

# 创建结果路径
os.makedirs(f'{args.result_path}', exist_ok=True)

dataset_root = args.dataset_root
iters = 24
epe_list = []

# 验证集推理
with torch.no_grad():
    # val_dataset = datasets.Canon(split='validation', root=os.path.join(dataset_root,'sony_all'))
    val_dataset = datasets.FlyingChairs4Img(split='validation')
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, _, _, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        # 进行前向传播并捕获特征图
        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)

        # 获取 EPE 误差
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        # 保存特征图和输入图像为 PNG 文件
        if 'hooked_layer' in features:
            # 保存第一个 batch 的特征图
            feature_maps = features['hooked_layer'][0]  # (C, H, W)
            plt.imshow(feature_maps[0].cpu().numpy(), cmap='viridis')
            # plt.colorbar()
            plt.savefig(f'{args.result_path}/feature_map_{val_id}_channel_{0}.png')
            plt.close()

        # 保存 image1
        image1_cpu = image1[0].cpu()
        image2_cpu = image2[0].cpu()
        concat_img = torch.concat([image1_cpu, image2_cpu], dim=-1)
        concat_img = concat_img.permute(1, 2, 0).byte().numpy()
        concat_img = Image.fromarray(concat_img).save(f'{args.result_path}/img_{val_id}.png')

    # 计算平均 EPE
    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)