# Dataloader for the DTU dataset in Yaoyao's format.
# by: Jiayu Yang
# date: 2020-01-28

# Note: This file use part of the code from the following projects.
#       Thanks for the authors for the great code.
#       MVSNet: https://github.com/YoYo000/MVSNet
#       MVSNet_pytorch: https://github.com/xy-guo/MVSNet_pytorch

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import random
from PIL import Image

from utils import read_pfm
from datasets.DTU.utils import read_cam_file

def scale_inputs(img, intrinsics, max_w, max_h, base=32):
    h, w = img.shape[:2]
    if h > max_h or w > max_w:
        scale = 1.0 * max_h / h
        if scale * w > max_w:
            scale = 1.0 * max_w / w
        new_w, new_h = scale * w // base * base, scale * h // base * base
    else:
        new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

    scale_w = 1.0 * new_w / w
    scale_h = 1.0 * new_h / h
    intrinsics[0, :] *= scale_w
    intrinsics[1, :] *= scale_h
    img = cv2.resize(img, (int(new_w), int(new_h)))
    return img, intrinsics


class DTU(Dataset):
    def __init__(self, config, mode):
        '''mode: train/val/test'''
        # Initializing the dataloader
        super(DTU, self).__init__()
        
        # Parse input
        self.config = config
        self.is_train = mode == 'train'
        self.robust_train = config['data']['robust_train'] if self.is_train and 'robust_train' in config['data'] else False
        self.lightings=range(7)
        self.mode = mode
        self.nsrcs = config['model']['params']['nsrc']
        self.ndepths = config['model']['params']['n_depths'] 

        self.data_root = os.path.join(config['data']['path'],
                                      'train' if self.mode in ('train', 'val') else 'test')
        self.scene_list = config['data']['tr_list'] if self.is_train else config['data']['te_list']
        with open(self.scene_list, 'r') as f:
            scene_names = f.readlines()
            self.scene_names = list(map(lambda x: x.strip(), scene_names))

        self.metas = self.build_list(self.is_train)

    def build_list(self, is_train):
        if self.mode == 'test':
            return self.build_test_list()

        metas = []
        pair_list = open('{}/Cameras/pair.txt'.format(self.data_root), 'r').readlines()

        pair_list = list(map(lambda x: x.strip(), pair_list))
        cnt = int(pair_list[0])
        for i in range(cnt):
            ref_id = int(pair_list[i*2+1])
            candidates = pair_list[i*2+2].split()
            src_ids = [int(candidates[2*j+1]) for j in range(self.nsrcs)]
            for scene_name in self.scene_names:
                for light in self.lightings:
                    metas.append([scene_name, ref_id, src_ids, light])
        return metas

    def build_test_list(self):
        metas = []
        for scene_name in self.scene_names:
            pair_list = open('{}/Cameras/pair.txt'.format(self.data_root), 'r').readlines()
            # pair_list = open('{}/{}/pair.txt'.format(self.data_root, scene_name), 'r').readlines()
            pair_list = list(map(lambda x: x.strip(), pair_list))
            cnt = int(pair_list[0])
            for i in range(cnt):
                ref_id = int(pair_list[i * 2 + 1])
                candidates = pair_list[i * 2 + 2].split()
                src_ids = [int(candidates[2 * j + 1]) for j in range(self.nsrcs)]
                metas.append([scene_name, ref_id, src_ids, 3])
        return metas

    def __len__(self):
        return len(self.metas)
    get_length = __len__

    def parse_cameras(self, path):
        cam_txt = open(path).readlines()
        f = lambda xs: list(map(lambda x: list(map(float, x.strip().split())), xs))

        extr_mat = f(cam_txt[1:5])
        intr_mat = f(cam_txt[7:10])

        extr_mat = np.array(extr_mat, np.float32)
        intr_mat = np.array(intr_mat, np.float32)

        min_dep, delta = list(map(float, cam_txt[11].strip().split()))
        max_dep = 1.06 * 191.5 * delta + min_dep

        if self.mode in ('train', 'val'):
            intr_mat[:2] *= 4.
            # note the loaded camera model is for 1/4 original resolution

        return extr_mat, intr_mat, min_dep, max_dep

    def parse_cameras_1(self, path):
        cam_txt = open(path).readlines()
        f = lambda xs: list(map(lambda x: list(map(float, x.strip().split())), xs))

        extr_mat = f(cam_txt[1:5])
        intr_mat = f(cam_txt[7:10])

        extr_mat = np.array(extr_mat, np.float32)
        intr_mat = np.array(intr_mat, np.float32)

        min_dep, max_dep = list(map(float, cam_txt[11].strip().split()))

        if self.mode in ('train', 'val'):
            intr_mat[:2] *= 4.
            # note the loaded camera model is for 1/4 original resolution

        return extr_mat, intr_mat, min_dep, max_dep

    def load_depths(self, path, scale):
        depth_s3 = np.array(read_pfm(path)[0], np.float32)
        depth_s3 = depth_s3 * scale
        h, w = depth_s3.shape
        depth_s2 = cv2.resize(depth_s3, (w//2, h//2), interpolation=cv2.INTER_LINEAR)
        depth_s1 = cv2.resize(depth_s3, (w//4, h//4), interpolation=cv2.INTER_LINEAR)
        return {'stage1': depth_s1, 'stage2': depth_s2, 'stage3': depth_s3}

    def make_masks(self, depths:dict, min_d, max_d):
        masks = {}
        for k, v in depths.items():
            m = np.ones(v.shape, np.uint8)
            m[v>max_d] = 0
            m[v<min_d] = 0
            masks[k] = m
        return masks

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scene_name, ref_view, src_views, light_idx = meta

        assert self.nsrcs <= len(src_views)

        # robust training strategy
        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nsrcs)
            src_views + [src_views[i] for i in index]
            scale = random.uniform(0.8, 1.25)
        else:
            scale = 1

        ref_img = [] 
        src_imgs = [] 
        ref_depths = []
        ref_depth_mask = [] 
        ref_intrinsics = [] 
        src_intrinsics = []
        ref_extrinsics = []
        src_extrinsics = []
        depth_min = []
        depth_max = []

        ## 1. Read images
        # ref image
        scene_name = '{}_train'.format(scene_name) if self.mode in ('train', 'val') else scene_name
        img_path = '{}/Rectified/{}/rect_{:03d}_{}_r5000.png'.format(self.data_root, scene_name, ref_view+1, light_idx)
        image = Image.open(img_path)
        ref_img = np.array(image, dtype=np.float32) / 255.
        # src image(s)
        for i in range(self.nsrcs):
            img_path = '{}/Rectified/{}/rect_{:03d}_{}_r5000.png'.format(self.data_root, scene_name, src_views[i]+1, light_idx)
            image = Image.open(img_path)
            src_imgs.append(np.array(image, dtype=np.float32) / 255.)


        ## 2. Read camera parameters
        cam_path = '{}/Cameras/train/{:08d}_cam.txt'.format(self.data_root, ref_view) if self.mode in ('train', 'val') \
            else '{}/Cameras/{:08d}_cam.txt'.format(self.data_root, ref_view)
        ref_extrinsics, ref_intrinsics, depth_min, depth_max = self.parse_cameras(cam_path)
        ref_extrinsics[:3,3] *= scale
        for i in range(self.nsrcs):
            cam_path = '{}/Cameras/train/{:08d}_cam.txt'.format(self.data_root, src_views[i]) if self.mode in ('train', 'val') \
                else '{}/Cameras/{:08d}_cam.txt'.format(self.data_root,  src_views[i])
            extrinsics, intrinsics, _, _ = self.parse_cameras(cam_path)
            extrinsics[:3,3] *= scale
            src_intrinsics.append(intrinsics)
            src_extrinsics.append(extrinsics)

        ## 3. Read Depth Maps
        if self.is_train or self.mode == 'val':
            # Read depth map of same size as input image first.
            depth_path = '{}/Depths_4/{}/depth_map_{:04d}.pfm'.format(self.data_root, scene_name, ref_view)
            ref_depths = self.load_depths(depth_path, scale)
            depth_min = depth_min*scale
            depth_max = depth_max*scale
            masks = self.make_masks(ref_depths, min_d=depth_min, max_d=depth_max)

        if self.mode == 'test':
            ref_img, ref_intrinsics = scale_inputs(ref_img, ref_intrinsics, 
                                                   max_h=self.config['data']['max_h'], 
                                                   max_w=self.config['data']['max_w'])
            src_imgs, src_intrinsics = zip(*[scale_inputs(img, intrinsics, 
                                                   max_h=self.config['data']['max_h'], 
                                                   max_w=self.config['data']['max_w'])
                    for img, intrinsics in zip(src_imgs, src_intrinsics)])
            

        # Orgnize output and return
        sample = {}
        sample["ref_img"] = np.moveaxis(np.array(ref_img),2,0)
        sample['src_imgs'] = np.moveaxis(np.array(src_imgs),3,1)
        sample["ref_in"] = np.array(ref_intrinsics)
        sample["src_in"] = np.array(src_intrinsics)
        sample["ref_ex"] = np.array(ref_extrinsics)
        sample["src_ex"] = np.array(src_extrinsics)
        sample['depth_range'] = np.array([depth_min, depth_max], np.float32)

        if self.is_train or self.mode == 'val' or self.mode == 'test':
            sample["fn"] = scene_name + '/{:0>8}'.format(ref_view)
            sample['ref_img_n'] = torch.from_numpy(np.ascontiguousarray(ref_img)) * 255
            sample['scene_name'] = scene_name
            sample["frame_idx"] = [ref_view] + src_views

        return sample
