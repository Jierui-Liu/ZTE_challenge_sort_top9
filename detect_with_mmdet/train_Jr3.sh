#! /bin/bash

config=configs/faster_rcnn_r50_fpn_1x.py
workdir=checkpoints/faster_rcnn_r50_fpn_1x_B_cut
# pretrained=pretrainmodels/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
pretrained=pretrainmodels/faster_rcnn_r50_fpn_1x_coco_pretrained_weights_classes_2.pth
# pretrained=checkpoints/faster_rcnn_r50_fpn_1x/epoch_6_81.889.pth


# config=configs/cascade_rcnn_r50_fpn_1x.py
# workdir=checkpoints/cascade_rcnn_r50_fpn_1x_B
# pretrained=pretrainmodels/cascade_rcnn_r50_fpn_20e_coco_pretrained_weights_classes_2.pth
# # pretrained=checkpoints/faster_rcnn_r50_fpn_1x/latest.pth


# config=configs/mask_rcnn_r50_fpn_1x.py
# workdir=checkpoints/mask_rcnn_r50_fpn_1x
# # pretrained=pretrainmodels/cascade_rcnn_r50_fpn_20e_coco_pretrained_weights_classes_2.pth
# pretrained=pretrainmodels/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth
# # pretrained=checkpoints/faster_rcnn_r50_fpn_1x/latest.pth
gpu_start=1
gpus=2
echo $pretrained
python tools/train.py $config \
                      --gpu_start $gpu_start --gpus $gpus \
                      --work_dir $workdir \
                      --pretrained_path $pretrained



                     
