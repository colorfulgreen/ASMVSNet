import argparse
import os
import numpy as np
import sys
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image
from typing import Dict, List, Tuple

sys.path.insert(0, '/data/src/ASMVSNet')
from utils import read_pfm
from datasets.DTU.utils import read_cam_file

def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


def read_pair_file(filename: str) -> List[Tuple[int, List[int]]]:
    """Read image pairs from text file and output a list of tuples each containing the reference image ID and a list of
    source image IDs
    Args:
        filename: pair text file path string
    Returns:
        List of tuples with reference ID and list of source IDs
    """
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        for _ in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) != 0:
                data.append((ref_view, src_views))
    return data

def read_image(filename: str, dim: Tuple[int, int] = (-1, -1)) -> Tuple[np.ndarray, int, int]:
    """Read image and rescale to specified dimensions (if exists)
    Args:
        filename: image input file path string
        dim: dimensions to scale down the image
    Returns:
        Tuple of scaled image along with original image height and width
    """
    image = Image.open(filename)
    # scale 0~255 to 0~1
    np_image = np.array(image, dtype=np.float32) / 255.0
    original_height = np_image.shape[0]
    original_width = np_image.shape[1]
    if dim[0] > 0:
        np_image = cv2.resize(np_image, dim, interpolation=cv2.INTER_LINEAR)

    return np_image, original_height, original_width

def save_image(filename: str, image: np.ndarray) -> None:
    """Save images including binary mask (bool), float (0<= val <= 1), or int (as-is)
    Args:
        filename: image output file path string
        image: output image array
    """
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    elif image.dtype == np.float32 or image.dtype == np.float64:
        image = image * 255
        image = image.astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    Image.fromarray(image).save(filename)

def scale_to_max_dim(image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, int, int]:
    """Scale image to specified max dimension
    Args:
        image: the input image in original size
        max_dim: the max dimension to scale the image down to if smaller than the actual max dimension
    Returns:
        Tuple of scaled image along with original image height and width
    """
    original_height = image.shape[0]
    original_width = image.shape[1]
    scale = max_dim / max(original_height, original_width)
    if 0 < scale < 1:
        width = int(scale * original_width)
        height = int(scale * original_height)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    return image, original_height, original_width

def read_map(path: str, max_dim: int = -1) -> np.ndarray:
    """ Read a binary depth map from either PFM or Colmap (bin) format determined by the file extension and also scale
    the map to the max dim if given
    Args:
        path: input depth map file path string
        max_dim: max dimension to scale down the map; keep original size if -1
    Returns:
        Array of depth map values
    """
    if path.endswith('.bin'):
        in_map = read_bin(path)
    elif path.endswith('.pfm'):
        in_map, _ = read_pfm(path)
    else:
        raise Exception('Invalid input format; only pfm and bin are supported')
    return scale_to_max_dim(in_map, max_dim)[0]


parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--geo_pixel_thres', type=float, default=1,
                    help='pixel threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres', type=float, default=0.01,
                    help='depth threshold for geometric consistency filtering')
parser.add_argument('--photo_thres', type=float, default=0.8, help='threshold for photometric consistency filtering')
parser.add_argument('--num_consis', type=int, default=3, help='consistency check for photometric consistency filtering')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

# project the reference point cloud into the source view, then project back
def reproject_with_depth(
    depth_ref: np.ndarray,
    intrinsics_ref: np.ndarray,
    extrinsics_ref: np.ndarray,
    depth_src: np.ndarray,
    intrinsics_src: np.ndarray,
    extrinsics_src: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project the reference points to the source view, then project back to calculate the reprojection error

    Args:
        depth_ref: depths of points in the reference view, of shape (H, W)
        intrinsics_ref: camera intrinsic of the reference view, of shape (3, 3)
        extrinsics_ref: camera extrinsic of the reference view, of shape (4, 4)
        depth_src: depths of points in the source view, of shape (H, W)
        intrinsics_src: camera intrinsic of the source view, of shape (3, 3)
        extrinsics_src: camera extrinsic of the source view, of shape (4, 4)

    Returns:
        A tuble contains
            depth_reprojected: reprojected depths of points in the reference view, of shape (H, W)
            x_reprojected: reprojected x coordinates of points in the reference view, of shape (H, W)
            y_reprojected: reprojected y coordinates of points in the reference view, of shape (H, W)
            x_src: x coordinates of points in the source view, of shape (H, W)
            y_src: y coordinates of points in the source view, of shape (H, W)
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    # step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    k_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = k_xyz_src[:2] / k_xyz_src[2:3]

    # step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    k_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = k_xyz_reprojected[:2] / k_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(
    depth_ref: np.ndarray,
    intrinsics_ref: np.ndarray,
    extrinsics_ref: np.ndarray,
    depth_src: np.ndarray,
    intrinsics_src: np.ndarray,
    extrinsics_src: np.ndarray,
    geo_pixel_thres: float,
    geo_depth_thres: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check geometric consistency and return valid points

    Args:
        depth_ref: depths of points in the reference view, of shape (H, W)
        intrinsics_ref: camera intrinsic of the reference view, of shape (3, 3)
        extrinsics_ref: camera extrinsic of the reference view, of shape (4, 4)
        depth_src: depths of points in the source view, of shape (H, W)
        intrinsics_src: camera intrinsic of the source view, of shape (3, 3)
        extrinsics_src: camera extrinsic of the source view, of shape (4, 4)
        geo_pixel_thres: geometric pixel threshold
        geo_depth_thres: geometric depth threshold

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            mask: mask for points with geometric consistency, of shape (H, W)
            depth_reprojected: reprojected depths of points in the reference view, of shape (H, W)
            x2d_src: x coordinates of points in the source view, of shape (H, W)
            y2d_src: y coordinates of points in the source view, of shape (H, W)
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)
    # print(depth_ref.shape)
    # print(depth_reprojected.shape)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(test_path, scan_folder, out_folder, plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh, num_consis):
    # the pair file
    pair_file = os.path.join(test_path, 'Cameras/pair.txt')
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    original_w = 1600
    original_h = 1200

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics, _, _ = read_cam_file(
            os.path.join(test_path, 'Cameras/{:0>8}_cam.txt'.format(ref_view)))
        ref_intrinsics[0] *= img_wh[0]/original_w
        ref_intrinsics[1] *= img_wh[1]/original_h
        # load the reference image
        # ref_img, _, _ = read_image(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)), max(img_wh))
        ref_img, _, _ = read_image(os.path.join(out_folder, 'rgb/{:0>8}.png'.format(ref_view)), img_wh)
        # load the estimated depth of the reference view
        ref_depth_est = read_map(os.path.join(out_folder, 'depth/dep_{:0>8}_3.pfm'.format(ref_view)))
        # ref_depth_est = np.squeeze(ref_depth_est, 2)
        # load the photometric mask of the reference view
        confidence = read_map(os.path.join(out_folder, 'confidence/conf_{:0>8}_3.pfm'.format(ref_view))) 

        photo_mask = confidence > photo_thres
        # photo_mask = np.squeeze(photo_mask, 2)

        all_srcview_depth_ests = []
        # compute the geometric mask
        geo_mask_sum = np.zeros_like(ref_depth_est, dtype=np.int32)
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics, _, _ = read_cam_file(
                os.path.join(test_path, 'Cameras/{:0>8}_cam.txt'.format(src_view)))
            src_intrinsics[0] *= img_wh[0]/original_w
            src_intrinsics[1] *= img_wh[1]/original_h
            # the estimated depth of the source view
            src_depth_est = read_map(os.path.join(out_folder, 'depth/dep_{:0>8}_3.pfm'.format(src_view)))

            geo_mask, depth_reprojected, _, _ = check_geometric_consistency(
                ref_depth_est, ref_intrinsics, ref_extrinsics, src_depth_est, src_intrinsics, src_extrinsics,
                geo_pixel_thres, geo_depth_thres)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        # large threshold, high accuracy, low completeness
        geo_mask = geo_mask_sum >= num_consis
        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_image(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_image(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_image(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)
        os.makedirs(os.path.join(out_folder, "depth_img"), exist_ok=True)

        print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(
            scan_folder, ref_view, geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

        if args.display:
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(1)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

        valid_points = final_mask
        # print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics), np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics), np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    img_wh = (1600, 1184)

    with open(args.testlist) as f:
        scans = f.readlines()
        scans = [line.rstrip() for line in scans]

    for scan in scans:
        scan_id = int(scan[4:])
        scan_folder = os.path.join(args.testpath, 'Rectified', scan)
        out_folder = os.path.join(args.outdir, scan)
        # step2. filter saved depth maps with geometric constraints
        filter_depth(args.testpath, scan_folder, out_folder, os.path.join(args.outdir, 'ucsnet{:0>3}_l3.ply'.format(scan_id)),
                     args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, args.num_consis)
