from os.path import join,dirname,realpath
import os
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
    return np.array(recordlist)

root_read='./demo/A-track/init_score0.4'
root_read='./demo/A-track/_75.738'
# root_read='/home/liujierui/proj/deep_sort_pytorch-master/demo/A-track/clean_insert_cleancn159_s0.3_nms0.3_yo_s4_10_s0.5_nms0.3_ds'
files=['1','4','5','9','10']
for file_name in files:
    root_save='/home/liujierui/proj/Dataset/A-track/test/MOT17-{:02}-FRCNN/det'.format(int(file_name))
    M=file2array(join(root_read,'Track'+file_name+'.txt'))
    print(join(root_read,'Track'+file_name+'.txt'),M.shape)
    M_new=np.zeros((len(M),7))
    M_new[:,1]=-1
    M_new[:,-1]=1
    M_new[:,2:6]=M[:,2:6]+0.0001
    M_new[:,0]=M[:,0]

    
    f = open(join(root_save,'det.txt'),'w')
    for l in M:
        f.write('{},-1,{:.3f},{:.3f},{:.3f},{:.3f},{:.5f}\n'\
                                .format(int(l[0]),l[2],l[3],l[4],l[5],l[6]))
    f.close()