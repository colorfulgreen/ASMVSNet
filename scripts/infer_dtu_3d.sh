#!/usr/bin/env bash

ROOT_PATH="/data/zhou/snap_dir/ASMVSNet/DTU/2024-03-19/sp"
DTU_TESTING="/data/datasets/dtu/test/"
TEST_LIST="../datasets/lists/dtu/test.txt"

python infer_3d.py -c ./configs/asmvsnet_dtu_sp.yaml

python fusion.py --dataset=dtu \
--geo_pixel_thres=0.1 --geo_depth_thres=0.01 --photo_thres 0. --num_consis 2 \
--testpath=$DTU_TESTING --outdir=$ROOT_PATH --testlist $TEST_LIST $@
