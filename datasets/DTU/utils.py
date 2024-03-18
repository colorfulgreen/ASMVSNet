import numpy as np
import re
import sys
from PIL import Image
import os, errno

def read_cam_file(filename):

    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1])
    depth_max = depth_min+(256*depth_interval)
    return intrinsics, extrinsics, depth_min, depth_max


def save_cameras2(extri, intri, path):
    cam_txt = open(path, 'w+')

    cam_txt.write('extrinsic\n')
    for i in range(4):
        for j in range(4):
            cam_txt.write(str(extri[i, j]) + ' ')
        cam_txt.write('\n')
    cam_txt.write('\n')

    cam_txt.write('intrinsic\n')
    for i in range(3):
        for j in range(3):
            cam_txt.write(str(intri[i, j]) + ' ')
        cam_txt.write('\n')
    cam_txt.close()
