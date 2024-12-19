import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder, BasicEncoder_64
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from extractor_denoise import get_model, get_enhance
from HFDB import HLFDB

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
            raise
        elif args.extractor_denoise:
            self.fnet = get_model(output_dim=hdim + cdim, pretrained=False)
            self.cnet = get_model(output_dim=hdim + cdim, pretrained=False)
        else:
            self.fnet = BasicEncoder_64(output_dim=256, norm_fn='instance', dropout=args.dropout)        
            self.cnet = BasicEncoder_64(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.f_feature_guide = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.c_feature_guide = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.feature_criterion = nn.MSELoss()  # 使用 MSE 作为 L2 损失

        if self.args.use_enhance:
            self.enhance = get_enhance()



    def feature_load_and_freeze(self):
        # Load pretrained weights from another RAFT model
        pretrained_weights = torch.load(self.args.feature_guide_ckpt)
        # 去除前缀 'module.' 
        # 去除前缀 'module.' 并提取 fnet 和 cnet 的权重
        fnet_weights = {k.replace('module.fnet.', ''): v for k, v in pretrained_weights.items() if k.startswith('module.fnet.')}
        cnet_weights = {k.replace('module.cnet.', ''): v for k, v in pretrained_weights.items() if k.startswith('module.cnet.')}

        # 将过滤后的权重加载到模型中
        self.f_feature_guide.load_state_dict(fnet_weights)
        self.c_feature_guide.load_state_dict(cnet_weights)
        # Freeze the weights of these layers
        for param in self.f_feature_guide.parameters():
            param.requires_grad = False

        for param in self.c_feature_guide.parameters():
            param.requires_grad = False
        print('load feature_guide_ckpt: ', self.args.feature_guide_ckpt)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1_ll, image2_ll, image1_nl, image2_nl, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        if self.args.use_enhance:
            image1_ll = self.enhance(image1_ll/255) * 255
            image2_ll = self.enhance(image2_ll/255) * 255

        image1_ll = 2 * (image1_ll / 255.0) - 1.0
        image2_ll = 2 * (image2_ll / 255.0) - 1.0
        image1_nl = 2 * (image1_nl / 255.0) - 1.0
        image2_nl = 2 * (image2_nl / 255.0) - 1.0

        image1_ll = image1_ll.contiguous()
        image2_ll = image2_ll.contiguous()
        image1_nl = image1_nl.contiguous()
        image2_nl = image2_nl.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1_ll = self.fnet(image1_ll)
            fmap2_ll = self.fnet(image2_ll)
            fmap1_nl = self.f_feature_guide(image1_nl)
            fmap2_nl = self.f_feature_guide(image2_nl)
                    
        
        fmap1_ll = fmap1_ll.float()
        fmap2_ll = fmap2_ll.float()
        fmap1_nl = fmap1_nl.float()
        fmap2_nl = fmap2_nl.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1_ll, fmap2_ll, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1_ll, fmap2_ll, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1_ll)
            cnet_nl = self.c_feature_guide(image1_nl)

            # 计算LD
            LD = self.feature_criterion(fmap1_ll, fmap1_nl) \
                    + self.feature_criterion(fmap2_ll, fmap2_nl) \
                    + self.feature_criterion(cnet, cnet_nl)


            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1_nl)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        
        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions, LD


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(16, 1, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.sigmoid(self.conv5(x))
        x = self.pool(x)
        return x.view(-1)
    
class CustomDiscriminatorLoss(nn.Module):
    def __init__(self, mu=0.5):
        super(CustomDiscriminatorLoss, self).__init__()
        self.mu = mu
        self.discriminator = Discriminator()
        self.mse = nn.MSELoss()

    def forward(self, fmap1_ll, fmap1_nl):
        lambda_D = self.discriminator(fmap1_ll)
        # D_nl = self.discriminator(fmap1_nl)
        d = self.mse(fmap1_ll, fmap1_nl)
        
        LD = lambda_D * (d**2) + (1 - lambda_D) * max((self.mu - d), 0)**2
        
        discriminator_loss = torch.mean(LD)
        
        return discriminator_loss
