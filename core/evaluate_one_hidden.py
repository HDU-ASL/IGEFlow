import torchvision
import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
from raft_hidden import RAFT
import datasets
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os

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
                   save_path):
    flow_pr_np = flow_pre[0].cpu()
    flow_gt_np = flow_gt
    image1 = image1[0].cpu()
    image2 = image2[0].cpu()

    flow_pr_img = torchvision.utils.flow_to_image(flow_pr_np)
    flow_gt_img = torchvision.utils.flow_to_image(flow_gt_np)

    image = transform(torch.concat([flow_pr_img, flow_gt_img], dim=-1))
    # Save the image
    image.save(save_path)

    concat_img = torch.concat([image1, image2], dim=-1)
    concat_img = concat_img.permute(1, 2, 0).byte().numpy()
    concat_img = Image.fromarray(concat_img).save(save_path.replace('flow_comparison','image'))

@torch.no_grad()
def validate_Canon(result_path, model, dataset_root, enhance, name, iters=24):
    os.makedirs(f'{result_path}/{name}_img', exist_ok=True)
    """ Perform evaluation on the Canon (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.Canon(split='validation', root=dataset_root)
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()


        _, flow_pr = model(image1, image2, image1.clone(), image2.clone(), iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())


        # save_results(flow_pr, flow_gt,
        #              image1, image2,
        #              f'{result_path}/{name}_img/flow_comparison_{val_id}.png')

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

@torch.no_grad()
def validate_chairs(model, iters=32):
    """ Perform evaluation on the FCDN (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all < 1)
    px3 = np.mean(epe_all < 3)
    px5 = np.mean(epe_all < 5)

    print("Validation Chairs EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe, px1, px3, px5))
    return {'chairs': epe}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='runs/debug/raft/raft-chairs.pth')
    parser.add_argument('--dataset', default='canon',help="dataset for evaluation")
    parser.add_argument('--result_path', default='runs/debug/raft')
    parser.add_argument('--dataset_root', default='/home/whx/code/Retinexformer/data/VBOF/VBOF_dataset/temp')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--extractor_denoise',action='store_true', default=False)
    parser.add_argument('--enhance',action='store_true', default=False)
    parser.add_argument('--vgg_loss',action='store_true', default=False)
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    os.makedirs(f'{args.result_path}', exist_ok=True)
    

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'canon':
            validate_Canon(args.result_path, model.module, args.dataset_root, args.enhance, 'sony_all')

        elif args.dataset == 'all':
            epe_fuji = validate_Canon(args.result_path, model.module, f'{args.dataset_root}/fuji_all', args.enhance, 'fuji_all')
            epe_sony = validate_Canon(args.result_path, model.module, f'{args.dataset_root}/sony_all', args.enhance, 'sony_all')
            epe_nikon = validate_Canon(args.result_path, model.module, f'{args.dataset_root}/nikon_all', args.enhance, 'nikon_all')
            epe_canon = validate_Canon(args.result_path, model.module, f'{args.dataset_root}/canon_all', args.enhance, 'canon_all')
            
            epe_combined = np.concatenate([epe_fuji, epe_sony, epe_nikon, epe_canon])
            epe_mean = np.mean(epe_combined)
            px1_combined = np.mean(epe_combined < 1)
            px3_combined = np.mean(epe_combined < 3)
            px5_combined = np.mean(epe_combined < 5)

            print("Combined Validation EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f" % (epe_mean, px1_combined, px3_combined, px5_combined))
            file_path = f"{args.result_path}/validation_all.txt"
            with open(file_path, "w") as file:
                file.write("Combined Validation EPE: %.3f, 1px: %.3f, 3px: %.3f, 5px: %.3f\n" % (epe_mean, px1_combined, px3_combined, px5_combined))