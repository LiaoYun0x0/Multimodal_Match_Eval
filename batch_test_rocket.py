import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import sys
import argparse
import random
import json
import numpy as np
import cv2
import time
import torch

from typing import Iterable, Optional
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, DistributedSampler)
import util
from models import build_model 
from datasets import build_dataset
from loss import build_criterion 
from common.logger import Logger, MetricLogger 
from common.functions import *
from configs import dynamic_load

mean_sar = np.array([0.33247536, 0.33247536, 0.33247536],dtype=np.float32).reshape(3,1,1)
std_sar = np.array([0.16769384, 0.16769384, 0.16769384],dtype=np.float32).reshape(3,1,1)
mean_opt = np.array([0.31578836, 0.31578836, 0.31578836],dtype=np.float32).reshape(3,1,1)
std_opt = np.array([0.1530546, 0.1530546 ,0.1530546],dtype=np.float32).reshape(3,1,1)

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global IM_POS
IM_POS = 0

@torch.no_grad()
def test(loader: Iterable, model: torch.nn.Module, print_freq=10000., tb_logger=None):
    model.eval()
    def _transform_inv(im, mean ,std):
        im = im * std + mean
        im  = np.uint8(im * 255.0)
        im = im.transpose(1,2,0)
        return im

    logger = MetricLogger(delimiter=' ')
    scores = 0
    i_err = {thr: 0 for thr in np.arange(1,11)}
    thres = [1,3,5,10]
    nums = 0
    dists_sa = []
    cost_time = 0

    for sample_batch in logger.log_every(loader, print_freq, header='Test'):
        scores+=1
        images1 = sample_batch["refer"].cuda().float()
        images0 = sample_batch["query"].cuda().float()
        gt_matrix=0
        H_gt = sample_batch['H_gt'].squeeze()

        t0 = time.time()
        preds = model(images0, images1, gt_matrix)
        t1 = time.time()
        cost_time = cost_time + (t1-t0)
        samples0 = _transform_inv(images0.detach().cpu().numpy().squeeze(), mean_sar, std_sar)
        samples1 = _transform_inv(images1.detach().cpu().numpy().squeeze(), mean_opt, std_opt)
        #out2 = draw_match_nir(preds['mkpts0'][:, 1:], preds['mkpts1'], samples0, samples1, 0, 0)

        #global IM_POS
        #cv2.imwrite(f"femip_rocket/{IM_POS}.jpg", out2)
        #IM_POS += 1
        i_err, num = eval_src_mma(preds['mkpts0'][:,1: ], preds['mkpts1'], samples0, samples1, i_err, H_gt)
        dist = eval_src_homography(preds['mkpts0'][:,1: ], preds['mkpts1'], samples0, samples1, H_gt)
        dists_sa.append(dist)
        nums += 1

    correct_sa = np.mean(
            [[float(dist <= t) for t in thres] for dist in dists_sa], axis=0)
    auc_sa = cal_error_auc(dists_sa, thresholds=thres)
    for thr in i_err:
        i_err[thr] = i_err[thr] / nums
    print(f"cost time: {cost_time / nums}")
    return i_err, auc_sa


def main(args):
    util.init_distributed_mode(args)
    seed = args.seed + util.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Seed used:', seed)

    model: torch.nn.Module = build_model(args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:', n_params)
    model = model.to(DEV)
    train_dataset, test_dataset = build_dataset(args)
    test_sampler = SequentialSampler(test_dataset)


    dataloader_kwargs = {
        #'collate_fn': train_dataset.collate_fn,
        'pin_memory': False,
        'num_workers': 0,
    }

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        drop_last=True,
        **dataloader_kwargs
    )
    res = {}
    model_names = os.listdir("/four_disk/Paper/old/old/Femip/loftr_rocket/artifacts/resnet101-dual_softmax_dim256-128_depth256-128")
    for model_name in [x for x in model_names if 'model_SAR2RGB_rotate_best_mean_std_mish_lbl_5230.6_108.4.pth'in x]:
        state_dict = torch.load(f"/four_disk/Paper/old/old/Femip/loftr_rocket/artifacts/resnet101-dual_softmax_dim256-128_depth256-128/{model_name}", map_location='cpu')
        model.load_state_dict(state_dict['model'])

        print(f'Start Testing model {model_name} ...')

        test_stats = test(test_loader,model)
        print(test_stats)
        res[model_name] = {'err':test_stats[0], 'auc': test_stats[1]}
    print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='imcnet_config')
    global_cfgs = parser.parse_args()

    args = dynamic_load(global_cfgs.config_name)
    print(args)
    print('=='*40 + '\n')

    main(args)
