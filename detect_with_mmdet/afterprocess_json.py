import json
import argparse
import os
from os.path import join,dirname,realpath
import numpy as np


if __name__ == '__main__':

    root_save='./Track_txt'
    read_filename='./result/result_A.bbox.json'
    # read_filename='./result/result_A_cascade.bbox.json'
    read_img_filename='/home/liujierui/proj/Dataset/annotations/track_A.json'
    with open(read_img_filename,'r') as load_f:
        load_dict = json.load(load_f)
    images_dict=load_dict['images']
    with open(read_filename,'r') as load_f:
        bboxs_dict = json.load(load_f)

    cnt=0
    last_vedio=1
    videos_num=[1,4,5,9,10]
    # videos_num=[str(i) for i in videos_num]
    M=[np.zeros((0,8)) for i in range(len(videos_num))]
    for bbox_dict in bboxs_dict:
        if bbox_dict['score']<=0.05:
            continue
        file_name=images_dict[bbox_dict['image_id']]['file_name'].split('/')
        frame_id=int(file_name[-1][:-4])
        video_id=int(file_name[0][5:])
        new=np.zeros((1,8))
        new[0,1]=cnt
        cnt=cnt+1
        new[0,0]=frame_id
        new[0,2:6]=np.array(bbox_dict['bbox'])
        new[0,6]=1
        new[0,7]=0
        video_index=videos_num.index(video_id)
        M[video_index]=np.vstack((M[video_index],new))
    try:
        shutil.rmtree(root_save)
    except:
        pass
    try:
        os.makedirs(root_save)
    except:
        pass
    for i in videos_num:
        video_index=videos_num.index(i)
        save_txt=root_save+'/Track{}.txt'.format(i)
        np.savetxt(save_txt,M[video_index].astype(int),fmt='%i', delimiter=",")
        print('/Track{}.txt'.format(i),'trandformed')
