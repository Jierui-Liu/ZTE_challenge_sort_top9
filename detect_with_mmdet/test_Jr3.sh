#! /bin/bash


# config2=configs/faster_rcnn_r50_fpn_1x.py
# # model_path2=checkpoints/faster_rcnn_r50_fpn_1x/latest.pth
# # model_path2=checkpoints/faster_rcnn_r50_fpn_1x/epoch_11_81.2.pth
# model_path2=checkpoints/faster_rcnn_r50_fpn_1x/epoch_3_81.909.pth
# # model_path2=checkpoints/faster_rcnn_r50_fpn_1x_i/epoch_4_82.173.pth
# # model_path2=checkpoints/faster_rcnn_r50_fpn_1x_i/epoch_3.pth
# out_filename2=./result/result_B
# # out_filename2=./result/result_A


# config2=configs/cascade_rcnn_r50_fpn_1x.py
# # model_path2=pretrainmodels/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth
# model_path2=checkpoints/cascade_rcnn_r50_fpn_1x/epoch_6.pth
# out_filename2=./result/result_A_cascade


# config2=configs/mask_rcnn_r50_fpn_1x.py
# model_path2=pretrainmodels/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth
# out_filename2=./result/result_B


config2=configs/faster_rcnn_r50_fpn_1x.py
# model_path2=checkpoints/faster_rcnn_r50_fpn_1x/latest.pth
# model_path2=checkpoints/faster_rcnn_r50_fpn_1x/epoch_11_81.2.pth
model_path2=checkpoints/faster_rcnn_r50_fpn_1x_B_cut/epoch_4.pth
# model_path2=checkpoints/faster_rcnn_r50_fpn_1x_i/epoch_4_82.173.pth
# model_path2=checkpoints/faster_rcnn_r50_fpn_1x_i/epoch_3.pth
out_filename2=./result/result_B
# out_filename2=./result/result_A

eval_mode=bbox
python tools/test.py $config2 \
                     $model_path2 \
                     --json_out $out_filename2 \
                     --eval $eval_mode
                     

python afterprocess_json.py