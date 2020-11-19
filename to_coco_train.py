import json
from os.path import join,dirname,realpath
import os
import cv2
import numpy as np
        
def file2array(path, delimiter=','):
    recordlist = []
    fp = open(path, 'r', encoding='utf-8')
    content = fp.read()     # content现在是一行字符串，该字符串包含文件所有内容
    fp.close()
    rowlist = content.splitlines()  # 按行转换为一维表，splitlines默认参数是‘\n’
    # 逐行遍历
    # 结果按分隔符分割为行向量
    recordlist = [[int(i) for i in row.split(delimiter)] for row in rowlist if row.strip()]
    M = np.array(recordlist)
    M = M[M[:,0].argsort(),:]  
    return M.astype(int)

root=dirname(realpath(__file__))
# txt_dir='rec_init_score0.8_nms0.3_0.85_crop_ensemble_more_fair0.4_0.5'
# txt_dir='after_i0c2_rec_mmdet_81.9_4_score0.7_iout0.55_B'
txt_dir='after_rec_dif_i0c2_B_rec_74.8_3_score0.7_iout0.55_B_75.592'
# txt_dir='after_rec_dif_i0instead_cut_c2_B_rec_74.8_3_score0.7_iout0.55_B_75.592'
# txt_dir='rec_c2_rec_mmdet_81.2_3_score0.8_iout0.55'
# txt_dir='rec_mmdet_11e_0.9_0_81.2'
# txt_dir='rec_c2_rec_mmdet_81.5_4_score0.8_iout0.55_82.173'
# txt_dir='rec_c2_rec_mmdet_81.5_4_score0.8_iout0.45'
# txt_dir='rec_new'
out_name=join(root,'track_B_train.json')
size={"Track1.txt":(1550,734),\
        "Track4.txt":(1920,980),\
        "Track5.txt":(1400,559),\
        "Track9.txt":(1116,874),\
        "Track10.txt":(615,593),\
        "Track2.txt":(1550,734),\
        "Track3.txt":(1116,874),\
        "Track6.txt":(1400,559),\
        "Track8.txt":(928,620),\
        "Track11.txt":(615,593),\
        "Track12.txt":(1728,824)}
json_out={}

json_out['images']=[]
json_out['annotations']=[]
# dataset='A-data'
dataset='B-data'
img_id=0
anno_id=0
for lists in os.listdir(join(root,dataset)): 
    path = os.path.join(join(root,dataset), lists)
    M=file2array(join(root,txt_dir,lists+'.txt'))
    s=size[lists+'.txt']
    frame_dict={}
    for f in range(1,M[-1,0]+1):
        frame_dict[f]={}#   以帧id为索引,物体id为次级索引

    for i in range(len(M)):
        l=M[i]
        frame_dict[l[0]][l[1]]=l.copy()

    for img_name in os.listdir(path): 
        img_dict={}
        # img=cv2.imread(join(path,img_name),0)
        # img_dict['file_name']=lists+'/img1/'+img_name
        img_dict['file_name']=lists+'/'+img_name
        img_dict['id']=img_id
        img_dict['height']=s[1]
        img_dict['width']=s[0]
        json_out['images'].append(img_dict)
        print('img_id',img_id)
        img_frame=int(img_name.split('.')[0])
        for o_key in frame_dict[img_frame].keys():
            anno={}
            # anno['segmentation']=[[float(frame_dict[img_frame][o_key][2]),float(frame_dict[img_frame][o_key][3]),\
            #                         float(frame_dict[img_frame][o_key][2])+1,float(frame_dict[img_frame][o_key][3]),\
            #                         float(frame_dict[img_frame][o_key][2])+1,float(frame_dict[img_frame][o_key][3])+1,\
            #                         float(frame_dict[img_frame][o_key][2]),float(frame_dict[img_frame][o_key][3])+1]]
            anno['image_id']=img_id
            anno['id']=anno_id
            anno['category_id']=1
            anno_id=anno_id+1
            anno['area']=frame_dict[img_frame][o_key][4]*frame_dict[img_frame][o_key][5]*1.0
            anno['bbox']=[float(frame_dict[img_frame][o_key][2]),float(frame_dict[img_frame][o_key][3]),\
                           float(frame_dict[img_frame][o_key][4]),float(frame_dict[img_frame][o_key][5])]
            json_out['annotations'].append(anno)
        
        img_id=img_id+1


json_out["categories"]= [{"supercategory": "person", "id": 1, "name": "person"}, {"supercategory": "vehicle", "id": 2, "name": "bicycle"}, {"supercategory": "vehicle", "id": 3, "name": "car"}, {"supercategory": "vehicle", "id": 4, "name": "motorcycle"}, {"supercategory": "vehicle", "id": 5, "name": "airplane"}, {"supercategory": "vehicle", "id": 6, "name": "bus"}, {"supercategory": "vehicle", "id": 7, "name": "train"}, {"supercategory": "vehicle", "id": 8, "name": "truck"}, {"supercategory": "vehicle", "id": 9, "name": "boat"}, {"supercategory": "outdoor", "id": 10, "name": "traffic light"}, {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"}, {"supercategory": "outdoor", "id": 13, "name": "stop sign"}, {"supercategory": "outdoor", "id": 14, "name": "parking meter"}, {"supercategory": "outdoor", "id": 15, "name": "bench"}, {"supercategory": "animal", "id": 16, "name": "bird"}, {"supercategory": "animal", "id": 17, "name": "cat"}, {"supercategory": "animal", "id": 18, "name": "dog"}, {"supercategory": "animal", "id": 19, "name": "horse"}, {"supercategory": "animal", "id": 20, "name": "sheep"}, {"supercategory": "animal", "id": 21, "name": "cow"}, {"supercategory": "animal", "id": 22, "name": "elephant"}, {"supercategory": "animal", "id": 23, "name": "bear"}, {"supercategory": "animal", "id": 24, "name": "zebra"}, {"supercategory": "animal", "id": 25, "name": "giraffe"}, {"supercategory": "accessory", "id": 27, "name": "backpack"}, {"supercategory": "accessory", "id": 28, "name": "umbrella"}, {"supercategory": "accessory", "id": 31, "name": "handbag"}, {"supercategory": "accessory", "id": 32, "name": "tie"}, {"supercategory": "accessory", "id": 33, "name": "suitcase"}, {"supercategory": "sports", "id": 34, "name": "frisbee"}, {"supercategory": "sports", "id": 35, "name": "skis"}, {"supercategory": "sports", "id": 36, "name": "snowboard"}, {"supercategory": "sports", "id": 37, "name": "sports ball"}, {"supercategory": "sports", "id": 38, "name": "kite"}, {"supercategory": "sports", "id": 39, "name": "baseball bat"}, {"supercategory": "sports", "id": 40, "name": "baseball glove"}, {"supercategory": "sports", "id": 41, "name": "skateboard"}, {"supercategory": "sports", "id": 42, "name": "surfboard"}, {"supercategory": "sports", "id": 43, "name": "tennis racket"}, {"supercategory": "kitchen", "id": 44, "name": "bottle"}, {"supercategory": "kitchen", "id": 46, "name": "wine glass"}, {"supercategory": "kitchen", "id": 47, "name": "cup"}, {"supercategory": "kitchen", "id": 48, "name": "fork"}, {"supercategory": "kitchen", "id": 49, "name": "knife"}, {"supercategory": "kitchen", "id": 50, "name": "spoon"}, {"supercategory": "kitchen", "id": 51, "name": "bowl"}, {"supercategory": "food", "id": 52, "name": "banana"}, {"supercategory": "food", "id": 53, "name": "apple"}, {"supercategory": "food", "id": 54, "name": "sandwich"}, {"supercategory": "food", "id": 55, "name": "orange"}, {"supercategory": "food", "id": 56, "name": "broccoli"}, {"supercategory": "food", "id": 57, "name": "carrot"}, {"supercategory": "food", "id": 58, "name": "hot dog"}, {"supercategory": "food", "id": 59, "name": "pizza"}, {"supercategory": "food", "id": 60, "name": "donut"}, {"supercategory": "food", "id": 61, "name": "cake"}, {"supercategory": "furniture", "id": 62, "name": "chair"}, {"supercategory": "furniture", "id": 63, "name": "couch"}, {"supercategory": "furniture", "id": 64, "name": "potted plant"}, {"supercategory": "furniture", "id": 65, "name": "bed"}, {"supercategory": "furniture", "id": 67, "name": "dining table"}, {"supercategory": "furniture", "id": 70, "name": "toilet"}, {"supercategory": "electronic", "id": 72, "name": "tv"}, {"supercategory": "electronic", "id": 73, "name": "laptop"}, {"supercategory": "electronic", "id": 74, "name": "mouse"}, {"supercategory": "electronic", "id": 75, "name": "remote"}, {"supercategory": "electronic", "id": 76, "name": "keyboard"}, {"supercategory": "electronic", "id": 77, "name": "cell phone"}, {"supercategory": "appliance", "id": 78, "name": "microwave"}, {"supercategory": "appliance", "id": 79, "name": "oven"}, {"supercategory": "appliance", "id": 80, "name": "toaster"}]
out_name=join(root,'track_B_train.json')

with open(out_name,"w") as f:
    json.dump(json_out,f)
    print(out_name+" 加载入文件完成...")