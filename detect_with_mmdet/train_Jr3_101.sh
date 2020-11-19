# #! /bin/bash

# config=configs/faster_rcnn_r101_fpn_1x.py
# workdir=checkpoints/faster_rcnn_r101_fpn_1x_cut
# # pretrained=pretrainmodels/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
# pretrained=pretrainmodels/faster_rcnn_r101_fpn_1x_coco_pretrained_weights_classes_2.pth
# # pretrained=checkpoints/faster_rcnn_r50_fpn_1x/epoch_6_81.889.pth


# gpu_start=3
# gpus=2
# echo $pretrained
# python tools/train.py $config \
#                       --gpu_start $gpu_start --gpus $gpus \
#                       --work_dir $workdir \
#                       --pretrained_path $pretrained







config2=configs/faster_rcnn_r101_fpn_1x.py
# model_path2=checkpoints/faster_rcnn_r50_fpn_1x/latest.pth
# model_path2=checkpoints/faster_rcnn_r50_fpn_1x/epoch_11_81.2.pth
model_path2=checkpoints/faster_rcnn_r101_fpn_1x_cut/epoch_3.pth
# model_path2=checkpoints/faster_rcnn_r50_fpn_1x_i/epoch_4_82.173.pth
# model_path2=checkpoints/faster_rcnn_r50_fpn_1x_i/epoch_3.pth
out_filename2=./result/result_101
# out_filename2=./result/result_A

eval_mode=bbox
python tools/test.py $config2 \
                     $model_path2 \
                     --json_out $out_filename2 \
                     --eval $eval_mode
                     
# python afterprocess_json.py