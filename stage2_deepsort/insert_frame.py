import os
from os.path import join,dirname,realpath
import numpy as np
def get_iou(box_xywh1,box_xywh2):
    b1=box_xywh1.copy()
    b1[2:]=box_xywh1[:2]+box_xywh1[2:]
    b2=box_xywh2.copy()
    b2[2:]=box_xywh2[:2]+box_xywh2[2:]
    delta_h = min(b1[2], b2[2]) - max(b1[0], b2[0])
    delta_w = min(b1[3], b2[3])-max(b1[1], b2[1])
    if delta_h < 0 or delta_w < 0:
        return 0
    else:
        overlap = delta_h * delta_w
        area = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - max(overlap, 0)
        iou = overlap / area
        return iou


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

def get_txtnames(path):
    out=[]
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename[-4:]=='.txt':
                out.append(filename)
    return out

label='init_score0.8_nms0.3_crop_motor'
dir_read='./demo/A-track/'+label
dir_save='./demo/A-track/insert_'+label
txts=get_txtnames(dir_read)
size={"Track1.txt":(1550,734),\
        "Track4.txt":(1920,980),\
        "Track5.txt":(1400,559),\
        "Track9.txt":(1116,874),\
        "Track10.txt":(615,593)}


try:
    shutil.rmtree(dir_save)
except:
    pass
try:
    os.makedirs(dir_save)
except:
    pass

for txt in txts:
    s=size[txt]
    area=s[0]*s[1]
    M=file2array(join(dir_read,txt))
    insert_dict={}
    object_dict={}
    frame_dict={}
    for f in range(1,M[-1,0]+1):
        insert_dict[f]={}#   以帧id为索引,物体id为次级索引
        frame_dict[f]={}#   以帧id为索引,物体id为次级索引
    
    for l in M:
        object_dict[l[1]]=[]#   以物体id为索引

    for i in range(len(M)):
        l=M[i]
        object_dict[l[1]].append(l[0])
        frame_dict[l[0]][l[1]]=l.copy()

    for o_key in object_dict.keys():
        if len(object_dict[o_key])==1:
            continue
        for i in range(len(object_dict[o_key][:-1])):
            start=object_dict[o_key][i]
            next=object_dict[o_key][i+1]
            iou_gap=get_iou(frame_dict[start][o_key][2:6],frame_dict[next][o_key][2:6])
            max_gap=30
            area_now=frame_dict[start][o_key][4]*frame_dict[start][o_key][5]
            if area_now>0.3*area:
                frame_dict[start].pop(o_key)
                continue
            if next==start+1 or next-start>max_gap or iou_gap<0.3/max_gap+0.05:
                continue
            for j in range(start+1,next):
                new=np.zeros(8).astype(int)
                new[0]=j
                new[1]=o_key
                new[-2]=1
                new[-1]=0
                box_start=frame_dict[start][o_key][2:6]
                box_next=frame_dict[next][o_key][2:6]
                ws=next-j
                we=j-start
                new[2:6]=(ws*box_start+we*box_next)/(next-start)
                overlap=False
                for o_key_f in frame_dict[j].keys():
                    iou=get_iou(new[2:6],frame_dict[j][o_key_f][2:6])
                    if iou>0.3:
                        overlap=True
                for o_key_f in insert_dict[j].keys():
                    iou=get_iou(new[2:6],insert_dict[j][o_key_f][2:6])
                    if iou>0.3:
                        overlap=True
                if overlap:
                    continue
                insert_dict[j][o_key]=new.astype(int)
        
        area_now=frame_dict[object_dict[o_key][-1]][o_key][4]*frame_dict[object_dict[o_key][-1]][o_key][5]
        if area_now>0.3*area:
            frame_dict[object_dict[o_key][-1]].pop(o_key)
    M_out=np.zeros((1,8))
    new=np.zeros((1,8))
    for f in range(1,M[-1,0]+1):
        for o_key in frame_dict[f].keys():
            new[0,:]=frame_dict[f][o_key].copy()
            M_out=np.vstack((M_out,new))
        
        for o_key in insert_dict[f].keys():
            new[0,:]=insert_dict[f][o_key].copy()
            M_out=np.vstack((M_out,new))
    M_out=M_out[1:]
    rate_cutoff=3/4
    M_out[:,3]=M_out[:,3]+M_out[:,5]*rate_cutoff/2
    M_out[:,5]=M_out[:,5]*(1-rate_cutoff)
    np.savetxt(join(dir_save,txt), M_out.astype(int),fmt='%i', delimiter=",")