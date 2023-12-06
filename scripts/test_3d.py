import os
import sys

sys.path.insert(0, '../')

import gc
import time
import argparse
import inspect
import random
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from tqdm import tqdm
import os.path as osp
import numpy as np
from PIL import Image
from datetime import datetime, date
from ruamel import yaml

from models.asmvsnet import ASMVSNet
from datasets.DTU.DTU import DTU
from utils import dict2numpy, mkdir_p, save_cameras2, write_pfm, tensor2cuda

torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_config(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.RoundTripLoader)
    return config

def parse_kwargs(minibatch, filtered_keys):
    kwargs = {k: v for k, v in minibatch.items() if k in filtered_keys}
    if torch.cuda.is_available():
        kwargs = tensor2cuda(kwargs)
    return kwargs

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def test_3d():
    parser = argparse.ArgumentParser(description='Evaluate script')
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    batch_size = 1
    world_size = 1

    args = parser.parse_args()

    os.chdir("..")
    is_main_process = True if args.local_rank == 0 else False

    config = load_config(args.config)
    model = ASMVSNet(**config["model"]["params"])
    state_dict = torch.load(config['model']['pretrained_model'], map_location='cuda:0')
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    filtered_keys = [p.name for p in inspect.signature(model.forward).parameters.values()]

    dataset = DTU(config=config, mode='test')
    batch_size = batch_size // world_size
    te_loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=batch_size * 1,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        sampler=None)
    niter_test = int(np.ceil(dataset.get_length() // batch_size))

    snap_dir = os.path.join(config['snap']['path'], config['model']['name'], config['data']['name'], date.today().isoformat(), '{}'.format(config['snap']['tag']))
    if is_main_process:
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niter_test), file=sys.stdout, bar_format=bar_format)
    else:
        pbar = range(niter_test)
    test_iter = iter(te_loader)

    abs_rel = 0.
    cnt = 0.

    for idx in pbar:
        minibatch = test_iter.next()
        scene_name = minibatch['scene_name'][0]
        scene_path = osp.join(snap_dir, scene_name)
        frame_idx = minibatch["frame_idx"][0][0]

        kwargs = parse_kwargs(minibatch, filtered_keys)
        with torch.no_grad():
            pred = model(**kwargs)
        outputs = dict2numpy(pred)

        rgb_path = osp.join(scene_path, 'rgb')
        mkdir_p(rgb_path)
        depth_path = osp.join(scene_path, 'depth')
        mkdir_p(depth_path)
        cam_path = osp.join(scene_path, 'cam')
        mkdir_p(cam_path)
        conf_path = osp.join(scene_path, 'confidence')
        mkdir_p(conf_path)

        ref_img = minibatch["ref_img"][0].numpy().transpose(1, 2, 0) * 255
        ref_img = np.clip(ref_img, 0, 255).astype(np.uint8)
        Image.fromarray(ref_img).save(rgb_path+'/{:08d}.png'.format(frame_idx))

        ref_ex = minibatch["ref_ex"][0].numpy()
        ref_in = minibatch["ref_in"][0].numpy()
        save_cameras2(ref_ex, ref_in, cam_path+'/cam_{:08d}.txt'.format(frame_idx))

        write_pfm(depth_path+"/dep_{:08d}_3.pfm".format(frame_idx), outputs['depth'][0,0])
        # write_pfm(conf_path+'/conf_{:08d}_3.pfm'.format(frame_idx), outputs['confidence'][0,0])

        if is_main_process:
            pbar.set_description('Saved results for {}/{} (resolution: {})'.format(scene_name, frame_idx, outputs['depth'].shape[0]))


        torch.cuda.empty_cache()

if __name__ == '__main__' :
    test_3d()
