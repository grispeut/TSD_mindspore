#!/bin/bash
python pascal2coco.py
rm -rf outputs
mpirun -n 4 python train.py --is_distributed 1 --per_batch_size 6 --T_max=100 --max_epoch=100 --warmup_epochs=4 --lr_scheduler=cosine_annealing
cp outputs/ckpt_0/0-100_*.ckpt weights/stage1.ckpt
rm -rf outputs
mpirun -n 4 python train.py --corrupt_aug --is_distributed 1 --per_batch_size 6 --T_max=320 --max_epoch=320 --warmup_epochs=4 --lr_scheduler=cosine_annealing --resume_yolov4 weights/stage1.ckpt
cp outputs/ckpt_0/0-320_*.ckpt weights/stage2.ckpt
rm -rf outputs
