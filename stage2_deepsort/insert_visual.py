import os
from os.path import join,dirname,realpath
import numpy as np
import cv2 as cv
import time

def file2array(path, delimiter=','):
    recordlist = []
    fp = open(path, 'r', encoding='utf-8')
    content = fp.read()     # content现在是一行字符串，该字符串包含文件所有内容
    fp.close()
    rowlist = content.splitlines()  # 按行转换为一维表，splitlines默认参数是‘\n’
    # 逐行遍历
    # 结果按分隔符分割为行向量
    recordlist = [[int(i) for i in row.split(delimiter)] for row in rowlist if row.strip()]
    return np.array(recordlist).astype(int)

def get_txtnames(path):
    out=[]
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename[-4:]=='.txt':
                out.append(filename)
    return out

# dir_todo='mmdet_11e_0.8'
dir_todo='after_delete_motor_mmdet_11e_0.9_iou0.5_0'
# dir_todo='init_score0.8_nms0.3_0.85_crop_ensemble_hrnet_fair0.3'
# dir_read='./demo/A-track/after_'+dir_todo
dir_read='./demo/A-track/'+dir_todo
# dir_read='./demo/A-track/rec_'+dir_todo
txts=get_txtnames(dir_read)
# dir_save='./demo/A-track/video_clean_insert'
# dir_save='./demo/A-track/video_clean_insert_clean'
# dir_save='./demo/A-track/video_rec'
# dir_save='./demo/A-track/video_mmdet'
dir_save='./demo/A-track/video_after'
# dir_save='./demo/A-track/video_clean_insert_merge'
root_img='/home/liujierui/proj/Dataset/A-data'
ids=[1,4,5,9,10]
size={"Track1.txt":(1550,734),\
        "Track4.txt":(1920,980),\
        "Track5.txt":(1400,559),\
        "Track9.txt":(1116,874),\
        "Track10.txt":(615,593)}
color=[(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0),(255,255,255)]
try:
    os.mkdir(dir_save)
except:
    pass

for txt in txts:
    # if not '10' in txt:
    #     continue
    print(txt)
    id=txt[:-4]
    id=id[5:]
    M=file2array(join(dir_read,txt))
    frame_dict={}
    a=M[-1,0]+1
    for f in range(1,M[-1,0]+1):
        frame_dict[f]={}#   以帧id为索引,物体id为次级索引

    for i in range(len(M)):
        l=M[i]
        frame_dict[l[0]][l[1]]=l.copy()

    path=join(root_img,'Track'+id,'img1')
    filelist_0 = os.listdir(path) #获取该目录下的所有文件名
    name=join(dir_save,id)
    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 15
    # size = (591,705) #图片的分辨率片
    file_path = name + ".avi"#导出路径
    # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')#不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    video = cv.VideoWriter(file_path, cv.VideoWriter_fourcc('M','J','P','G'), fps, size[txt])
 
    # video = cv2.VideoWriter( file_path, fourcc, fps, size )
    filelist=[str(i) for i in range(1,len(filelist_0)+1)]
    for id_img in filelist:

        item = path + '/' + id_img+'.jpg' 
        img = cv.imread(item)  #使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
        if int(id_img) in frame_dict.keys():
            if len(frame_dict[int(id_img)].keys())>0:
                for o_key in frame_dict[int(id_img)].keys():
                    box=frame_dict[int(id_img)][o_key][2:6]
                    img = cv.putText(img, str(o_key), (box[0],box[1]), cv.FONT_HERSHEY_SIMPLEX, 1.2, color[o_key%len(color)], 2)
                    img = cv.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]),color[o_key%len(color)] , 1)
        video.write(img)        #把图片写进视频
 
    video.release() #释放

    