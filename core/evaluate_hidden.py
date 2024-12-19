import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from raft_hidden import RAFT
from utils.utils import InputPadder, forward_interpolate
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

def flow_to_image(flow):
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    angle = np.arctan2(flow[..., 1], flow[..., 0])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1
    hsv[..., 2] = magnitude / np.max(magnitude)
    
    rgb = plt.cm.hsv(hsv[..., 0])[:, :, :3] * 255
    rgb = rgb.astype(np.uint8)
    
    return Image.fromarray(rgb)

# Convert the tensor to a PIL Image
transform = transforms.ToPILImage()
def save_results(flow_pre, flow_gt,
                 image1, image2,
                 temp1, temp2,
                   save_path):
    flow_pr_np = flow_pre[0].cpu()
    flow_gt_np = flow_gt
    image1 = image1[0].cpu()
    image2 = image2[0].cpu()
    temp1 = temp1[0].cpu()
    temp2 = temp2[0].cpu()


    flow_pr_img = torchvision.utils.flow_to_image(flow_pr_np)
    flow_gt_img = torchvision.utils.flow_to_image(flow_gt_np)

    image = transform(torch.concat([flow_pr_img, flow_gt_img], dim=-1))
    # Save the image
    image.save(save_path)

    concat_img = torch.concat([image1, image2], dim=-1)
    concat_img = concat_img.permute(1, 2, 0).byte().numpy()
    concat_img = Image.fromarray(concat_img).save(save_path.replace('flow_comparison','image'))

    concat_img = torch.concat([temp1, temp2], dim=-1)
    concat_img = concat_img.permute(1, 2, 0).byte().numpy()
    concat_img = Image.fromarray(concat_img).save(save_path.replace('flow_comparison','temp'))


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs4Img(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, image1_H, image2_H, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        image1_H = image1_H[None].cuda()
        image2_H = image2_H[None].cuda()

        _, flow_pr = model(image1, image2, image1_H, image2_H, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}

@torch.no_grad()
def validate_Canon(model, iters=24):
    """ Perform evaluation on the Canon (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.Canon(split='validation', root='/home/whx/code/Retinexformer/data/VBOF/VBOF_dataset/all')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()


        _, flow_pr = model(image1, image2, image1.clone(), image2.clone(), iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'sony': epe}


@torch.no_grad()
def validate_VBOF(result_path, model, dataset_root, use_enhance, name, iters=24):
    os.makedirs(f'{result_path}/{name}_img', exist_ok=True)
    """ Perform evaluation on the Canon (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.Canon(split='validation', root=os.path.join(dataset_root,name))
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()


        _, flow_pr, temp1, temp2 = model(image1, image2, image1.clone(), image2.clone(), iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())


        save_results(flow_pr, flow_gt,
                     image1, image2,
                    temp1, temp2,
                     f'{result_path}/{name}_img/flow_comparison_{val_id}.png')

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    file_path = f"{result_path}/validation_{name}.txt"
    with open(file_path, "w") as file:
        file.write("Validation EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f\n" % (epe, px1, px3, px5))
    # return {'canon': epe}
    return epe_all






def fun(result_path, model, dataset_root, use_enhance, name, iters=24):
    os.makedirs(f'{result_path}/{name}_img', exist_ok=True)
    """ Perform evaluation on the Canon (test) split """
    model.eval()
    epe_list = []
    val_dataset = datasets.FlyingChairs4Img(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, image1_H, image2_H, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        image1_H = image1_H[None].cuda()
        image2_H = image2_H[None].cuda()

        # _, flow_pr, temp1, temp2 = model(image1, image2, image1_H, image2_H, iters=iters, test_mode=True)
        _, flow_pr, temp1, temp2= model(image1, image2, image1_H, image2_H, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        save_results(flow_pr, flow_gt,
                     image1, image2,
                     temp1, temp2,
                     f'{result_path}/{name}_img/flow_comparison_{val_id}.png')

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    file_path = f"{result_path}/validation_{name}.txt"
    with open(file_path, "w") as file:
        file.write("Validation EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f\n" % (epe, px1, px3, px5))
    # return {'canon': epe}
    return epe_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='runs/raw_4img_origin_lyt/checkpoints/55000_raw_4img_origin_lyt.pth')
    # parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--use_enhance', action='store_true', default=False)
    parser.add_argument('--result_path', default='runs/debug')
    parser.add_argument('--extractor_denoise', action='store_true', help='use extractor_denoise model')
    parser.add_argument('--dataset_root', default='/home/whx/code/Retinexformer/data/VBOF/VBOF_dataset')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model))
    state_dict = torch.load(args.model)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if k.startswith('module.'):
    #         new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
    #     else:
    #         new_state_dict[k] = v
    model.load_state_dict(state_dict)



    model.cuda()
    model.eval()

    os.makedirs(f'{args.result_path}', exist_ok=True)
    # 3468 - 233MB
    # import time
    with torch.no_grad():
        start_time = time.time()
        epe_fuji = fun(args.result_path, model.module, f'{args.dataset_root}', args.use_enhance, 'canon_all')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.4f} seconds")
        raise

        epe_fuji = validate_VBOF(args.result_path, model.module, f'{args.dataset_root}', args.use_enhance, 'sony_all')
        epe_sony = validate_VBOF(args.result_path, model.module, f'{args.dataset_root}', args.use_enhance, 'fuji_all')
        epe_nikon = validate_VBOF(args.result_path, model.module, f'{args.dataset_root}', args.use_enhance, 'nikon_all')
        epe_canon = validate_VBOF(args.result_path, model.module, f'{args.dataset_root}', args.use_enhance, 'canon_all')
        
        epe_combined = np.concatenate([epe_fuji, epe_sony, epe_nikon, epe_canon])
        epe_mean = np.mean(epe_combined)
        px1_combined = np.mean(epe_combined < 1)
        px3_combined = np.mean(epe_combined < 3)
        px5_combined = np.mean(epe_combined < 5)

        print("Combined Validation EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe_mean, px1_combined, px3_combined, px5_combined))
        file_path = f"{args.result_path}/validation_all.txt"
        with open(file_path, "w") as file:
            file.write("Combined Validation EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f\n" % (epe_mean, px1_combined, px3_combined, px5_combined))

