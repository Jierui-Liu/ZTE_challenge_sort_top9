import os
from os.path import join,dirname,realpath
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

def get_txtnames(path):
    out=[]
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename[-4:]=='.txt':
                out.append(filename)
    return out

# label='init_score0.8_nms0.3_0.85_crop_ensemble_more_fair0.4_0.5'
# label='init_score0.8_nms0.3_0.85_crop_ensemble_hrnet_fair0.3'
# label='mmdet_rec_81.4_10_score0.9_iout0.55'
label='c2_rec_mmdet_81.2_3_score0.8_iout0.55'
dir_read='./demo/A-track/'+label
dir_save='./demo/A-track/rec_'+label
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
rate_hw_max=3.5
rate_hw_min=1
to_rec=['Track10.txt']

for txt in txts:
    cnt=0
    cnt1=0
    cnt2=0
    s=size[txt]
    area=s[0]*s[1]
    M=file2array(join(dir_read,txt))
    frame_dict={}
    for f in range(1,M[-1,0]+1):
        frame_dict[f]={}#   以帧id为索引,物体id为次级索引
    
    for i in range(len(M)):
        l=M[i]
        left=l[2]
        top=l[3]
        w=l[4]
        h=l[5]
        # if not '10' in txt:
        #     frame_dict[l[0]][l[1]]=l.copy()
        #     continue

        if True:#txt in to_rec:
            if h*1.0/w>rate_hw_max:
                cnt=cnt+1
                x_center=left+w/2
                w_new=h/rate_hw_max
                l[2]=x_center-w_new/2
                l[4]=w_new
            elif h*1.0/w<rate_hw_min:
                cnt1=cnt1+1
                x_center=left+w/2
                if w>s[0]/6:
                    w_new=s[0]/6
                    l[2]=x_center-w_new/2
                    l[4]=w_new
                else:
                    w_new=h/rate_hw_min
                    l[2]=x_center-w_new/2
                    l[4]=w_new
            elif w>s[0]/6:
                cnt2=cnt2+1
                x_center=left+w/2
                w_new=s[0]/6
                l[2]=x_center-w_new/2
                l[4]=w_new
            
        frame_dict[l[0]][l[1]]=l.copy()

    
    M_out=np.zeros((1,8))
    new=np.zeros((1,8))
    for f in range(1,M[-1,0]+1):
        for o_key in frame_dict[f].keys():
            new[0,:]=frame_dict[f][o_key].copy()
            M_out=np.vstack((M_out,new))
        
    M_out=M_out[1:]
    np.savetxt(join(dir_save,txt), M_out.astype(int),fmt='%i', delimiter=",")
    print(txt,cnt,cnt1,cnt2)