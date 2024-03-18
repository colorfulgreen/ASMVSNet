import os
import sys
import logging
import inspect
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import os.path as osp
import numpy as np
from PIL import Image
from datetime import date
from ruamel import yaml 

sys.path.insert(0, '/data/src/ASMVSNet')

from net.ASMVSNet import ASMVSNet 
from datasets.DTU import DTU, save_cameras2

from utils import mkdir_p, write_pfm, dict2numpy, tensor2cuda

torch.benchmark = False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_config(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.RoundTripLoader)
    return config

def load_pretrained_model(model, path):
    logging.info('loadding pretrained model from {}.'.format(path))
    state_dict = torch.load(path, map_location='cuda:0')
    if 'model' in state_dict.keys():
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=False)

    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logging.warning('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))
    if len(unexpected_keys) > 0:
        logging.warning('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))


def infer_3d():
    parser = argparse.ArgumentParser(description='Inference 3d')
    parser.add_argument('-c', '--config', type=str)
    args = parser.parse_args()

    os.chdir("..")

    config = load_config(args.config)
    model = ASMVSNet(**config["model"]["params"])
    if config['model'].get('pretrained_model') is not None:
        load_pretrained_model(model, config['model']['pretrained_model'])
    model.cuda()
    model.eval()

    filtered_keys = [p.name for p in inspect.signature(model.forward).parameters.values()]

    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    batch_size = world_size
    dataset = DTU(config=config, mode='test')
    dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=batch_size * 1,
                        drop_last=False,
                        shuffle=False,
                        pin_memory=True,
                        sampler=None)
    dataloader_iter = iter(dataloader)
    n_iters = int(np.ceil(dataset.get_length() // batch_size))

    snap_dir = os.path.join(config['snap']['path'], config['model']['name'], config['data']['name'], date.today().isoformat(), '{}'.format(config['snap']['tag']))

    pbar = tqdm(range(n_iters), file=sys.stdout, 
                bar_format='{desc}[{elapsed}<{remaining},{rate_fmt}]')
    for idx in pbar:
        minibatch = dataloader_iter.next()
        scene_name = minibatch['scene_name'][0]
        scene_path = osp.join(snap_dir, scene_name)
        frame_idx = minibatch["frame_idx"][0][0]

        with torch.no_grad():
            kwargs = {k: v for k, v in minibatch.items() if k in filtered_keys}
            if torch.cuda.is_available():
                kwargs = tensor2cuda(kwargs)
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
        write_pfm(conf_path+'/conf_{:08d}_3.pfm'.format(frame_idx), outputs['confidence'][0,0])

        pbar.set_description('{}/{} (resolution: {})'.format(scene_name, frame_idx, outputs['depth'].shape))


if __name__ == '__main__' :
    infer_3d()
