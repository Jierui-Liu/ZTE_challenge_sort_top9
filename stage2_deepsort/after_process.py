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
        
def get_area_iou(box_xywh1,box_xywh2):
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
        return overlap


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


def after_process(dir_txt,txt,s,max_gap=20):
    rate_hw_max=3.5
    rate_hw_min=1
    difficult_txt=['Track1.txt','Track5.txt','Track9.txt','Track4.txt']
    # if '1.txt' not in txt:
    #     continue
    # if not '4' in txt:
    #     continue
    # 建立存储结构体
    area=s[0]*s[1]
    M=file2array(join(dir_txt,txt))
    insert_dict={}
    object_dict={}
    frame_dict={}
    for f in range(1,M[-1,0]+1):
        insert_dict[f]={}#   以帧id为索引,物体id为次级索引
        frame_dict[f]={}#   以帧id为索引,物体id为次级索引
    
    for l in M:
        object_dict[l[1]]=[]#   以物体id为索引
    cnt=0
    cnt1=0
    cnt2=0
    for i in range(len(M)):
        l=M[i]
        left=l[2]
        top=l[3]
        w=l[4]
        h=l[5]
        
        if True:#txt=='Track10.txt':
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
            
        object_dict[l[1]].append(l[0])
        frame_dict[l[0]][l[1]]=l.copy()
    print(txt,cnt,cnt1,cnt2)


    # # 删除数量太少的帧
    # to_delete=[]
    # cnt=0
    # for o_key in object_dict.keys():
    #     num=len(object_dict[o_key])
    #     if num<10:
    #         a=object_dict[o_key]
    #         cnt=cnt+1
    #         for f in object_dict[o_key]:
    #             frame_dict[f].pop(o_key)
    #         to_delete.append(o_key)
    # for o_key in to_delete:
    #         object_dict.pop(o_key)

    if txt in difficult_txt:
        # 高重复的高处帧
        cnt=0
        for f in range(1,M[-1,0]+1):
            to_delete=[]
            for o_key in frame_dict[f].keys():
                for other_o_key in frame_dict[f].keys():
                    if o_key==other_o_key:
                        continue
                    box_now=frame_dict[f][o_key][2:6]
                    box_other=frame_dict[f][other_o_key][2:6]
                    area_iou=get_area_iou(box_now,box_other)*1.0
                    area_now=box_now[2]*box_now[3]*1.0
                    area_other=box_other[2]*box_other[3]*1.0
                    y_now=box_now[1]+box_now[3]/2
                    y_other=box_other[1]+box_other[3]/2
                    if  area_iou/area_now>0.9 and\
                         area_other<0.1*area and not(o_key in to_delete):
                        to_delete.append(o_key)
            # print('txt:',txt,'frame_id:',f,'to_delete:',to_delete)
            # if len(to_delete)>0:
            #     a=1
            for o_key in to_delete:
                frame_dict[f].pop(o_key) 
                object_dict[o_key].remove(f)

    # 插帧
    to_delete=[]
    for o_key in object_dict.keys():
        if len(object_dict[o_key])==1:
            continue
        elif len(object_dict[o_key])==0:
            to_delete.append(o_key)
            continue
        for i in range(len(object_dict[o_key][:-1])):
            start=object_dict[o_key][i]
            next=object_dict[o_key][i+1]
            iou_gap=get_iou(frame_dict[start][o_key][2:6],frame_dict[next][o_key][2:6])
            area_now=frame_dict[start][o_key][4]*frame_dict[start][o_key][5]
            if area_now>0.3*area:
                frame_dict[start].pop(o_key)
                continue
            if next==start+1 or next-start>max_gap or iou_gap<0.3/max_gap+0.1:
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
                    if iou>0.6:
                        overlap=True
                        pass
                # for o_key_f in insert_dict[j].keys():
                #     iou=get_iou(new[2:6],insert_dict[j][o_key_f][2:6])
                #     if iou>0.4:
                #         overlap=True
                #         pass
                if overlap:
                    continue
                insert_dict[j][o_key]=new.astype(int)
        a=object_dict[o_key]
        area_now=frame_dict[object_dict[o_key][-1]][o_key][4]*frame_dict[object_dict[o_key][-1]][o_key][5]
        if area_now>0.3*area:
            frame_dict[object_dict[o_key][-1]].pop(o_key)

    for o_key in to_delete:
        object_dict.pop(o_key)

    M_mid=np.zeros((1,8))
    new=np.zeros((1,8))
    for f in range(1,M[-1,0]+1):
        for o_key in frame_dict[f].keys():
            new[0,:]=frame_dict[f][o_key].copy()
            M_mid=np.vstack((M_mid,new))
        
        for o_key in insert_dict[f].keys():
            new[0,:]=insert_dict[f][o_key].copy()
            M_mid=np.vstack((M_mid,new))
    M_mid=M_mid[1:].astype(int)
    frame_dict={}
    for f in range(1,M_mid[-1,0]+1):
        frame_dict[f]={}#   以帧id为索引,物体id为次级索引
    

    for i in range(len(M_mid)):
        l=M_mid[i]
        frame_dict[l[0]][l[1]]=l.copy()
        
    if txt in difficult_txt:
        # 高重复的高处帧
        cnt=0
        for f in range(1,M[-1,0]+1):
            to_delete=[]
            for o_key in frame_dict[f].keys():
                for other_o_key in frame_dict[f].keys():
                    if o_key==other_o_key:
                        continue
                    box_now=frame_dict[f][o_key][2:6]
                    box_other=frame_dict[f][other_o_key][2:6]
                    area_iou=get_area_iou(box_now,box_other)*1.0
                    area_now=box_now[2]*box_now[3]*1.0
                    area_other=box_other[2]*box_other[3]*1.0
                    y_now=box_now[1]+box_now[3]/2
                    y_other=box_other[1]+box_other[3]/2
                    if  area_iou/area_now>0.9 and\
                         area_other<0.2*area and not(o_key in to_delete):
                        to_delete.append(o_key)
            # print('txt:',txt,'frame_id:',f,'after_to_delete:',to_delete)
            # if len(to_delete)>0:
            #     a=1
            for o_key in to_delete:
                frame_dict[f].pop(o_key) 
                a=frame_dict[f]

    # M_out=np.zeros((1,8))
    # new=np.zeros((1,8))
    # for f in range(1,M[-1,0]+1):
    #     for o_key in frame_dict[f].keys():
    #         new[0,:]=frame_dict[f][o_key].copy()
    #         M_out=np.vstack((M_out,new))

    # M_out=M_out[1:].astype(int)
    # object_dict={}
    # frame_dict={}
    # for f in range(1,M[-1,0]+1):
    #     insert_dict[f]={}#   以帧id为索引,物体id为次级索引
    #     frame_dict[f]={}#   以帧id为索引,物体id为次级索引

    # for l in M_out:
    #     object_dict[l[1]]=[]#   以物体id为索引

    # for i in range(len(M_out)):
    #     l=M_out[i]
    #     object_dict[l[1]].append(l[0])
    #     frame_dict[l[0]][l[1]]=l.copy()

    # return frame_dict,M_out
    return frame_dict,M_mid
        

if __name__=="__main__":
    # label='c2_rec_mmdet_81.5_4_score0.8_iout0.55_82.173'
    label='new'
    dir_read='./demo/A-track/'+label
    dir_save='./demo/A-track/after_ci'+label
    txts=get_txtnames(dir_read)
    size={"Track1.txt":(1550,734),\
            "Track4.txt":(1920,980),\
            "Track5.txt":(1400,559),\
            "Track9.txt":(1116,874),\
            "Track10.txt":(615,593)}

    difficult_txt=['Track1.txt','Track5.txt','Track9.txt','Track4.txt']

    try:
        shutil.rmtree(dir_save)
    except:
        pass
    try:
        os.makedirs(dir_save)
    except:
        pass

    for txt in txts:
        print(txt)
        frame_dict,M_out=after_process(dir_read,txt,size[txt],max_gap=45)
        np.savetxt(join(dir_save,txt), M_out.astype(int),fmt='%i', delimiter=",")
