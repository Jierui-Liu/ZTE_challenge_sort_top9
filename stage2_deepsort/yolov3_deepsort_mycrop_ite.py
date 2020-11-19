import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from os.path import join,dirname,realpath
from transform.stainNorm_Reinhard import Normalizer
from detector.YOLOv3.nms.nms import *

def get_avinames(path):
    out=[]
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename[-4:]=='.avi':
                out.append(filename)
    return out
 
def file2array(path, delimiter=','):
    recordlist = []
    fp = open(path, 'r', encoding='utf-8')
    content = fp.read()     # content现在是一行字符串，该字符串包含文件所有内容
    fp.close()
    rowlist = content.splitlines()  # 按行转换为一维表，splitlines默认参数是‘\n’
    # 逐行遍历
    # 结果按分隔符分割为行向量
    recordlist = [[float(i) for i in row.split(delimiter)] for row in rowlist if row.strip()]
    M = np.array(recordlist)
    M = M[M[:,0].argsort(),:]  
    return M



class VideoTracker(object):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.use_cuda = args.use_cuda and torch.cuda.is_available()
        if not self.use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=self.use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=self.use_cuda)
        self.class_names = self.detector.class_names

    def sort_init(self):
        self.deepsort = build_tracker(self.cfg, use_cuda=self.use_cuda)

    def __enter__(self):
        pass
    #     if self.args.cam != -1:
    #         ret, frame = self.vdo.read()
    #         assert ret, "Error: Camera error"
    #         self.im_width = frame.shape[0]
    #         self.im_height = frame.shape[1]

    #     else:
    #         assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
    #         self.vdo.open(self.args.VIDEO_PATH)
    #         self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
    #         self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         assert self.vdo.isOpened()

    #     if self.args.save_path:
    #         # fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
    #         self.writer = cv2.VideoWriter(self.args.save_path, cv2.VideoWriter_fourcc('M','J','P','G'), 12, (self.im_width,self.im_height))

        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
    #     if exc_type:
    #         print(exc_type, exc_value, exc_traceback)


    def run(self,id_video):
        Normalizer_my=Normalizer()
        im_target=cv2.imread('/home/liujierui/proj/Dataset/Target_StainNorm/535.jpg')
        Normalizer_my.fit(im_target)

        self.sort_init()
        try:
            os.makedirs(self.args.save_path)
        except:
            pass
        idx_frame = 0
        txt_name=join(self.args.save_path,'Track'+str(id_video)+'.txt')
        # path=join(self.args.VIDEO_PATH,'Track'+str(id_video),'img1')    #把img去掉
        path=join(self.args.VIDEO_PATH,'Track'+str(id_video))    #把img去掉
        filelist_0 = os.listdir(path) #获取该目录下的所有文件名
        filelist=[str(i) for i in range(1,len(filelist_0)+1)]   
        avi_name = join(self.args.save_path,str(id_video) + ".avi")#导出路径

        # dirtxt_read='/home/liujierui/proj/deep_sort_pytorch-master/demo/A-track/cut_c2_B_rec_74.8_3_score0.7_iout0.55_B_75.592'
        dirtxt_read='/home/liujierui/proj/mmdetection/Track_txt'
        txt_read=join(dirtxt_read,'Track'+str(id_video)+'.txt')
        M=file2array(txt_read)

        frame_dict={}
        for f in range(1,int(M[-1,0])+1):
            frame_dict[f]={}#   以帧id为索引,物体id为次级索引
        for i in range(len(M)):
            l=M[i]
            frame_dict[int(l[0])][int(l[1])]=l.copy()

        item = path + '/1.jpg' 
        ori_im = cv2.imread(item)
        video = cv2.VideoWriter(avi_name, cv2.VideoWriter_fourcc('M','J','P','G'), 15, (ori_im .shape[1],ori_im.shape[0]))
        
        M=np.zeros((1,8))
        for id_img in filelist:
            item = path + '/' + id_img+'.jpg' 
            ori_im = cv2.imread(item)
            # ori_im=Normalizer_my.transform(ori_im)
            # ori_im=ori_im.astype(np.uint8)
            idx_frame += 1

            start = time.time()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            bbox_xywh=None
            cls_conf=None
            cls_ids=None

            
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids==0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

            id_int=int(id_img)
            if len(frame_dict[id_int].keys())==0:
                bbox_xywh_other=None
            else:
                bbox_xywh_other=np.zeros((0,4))
                new=np.zeros((1,4))
                cls_conf_other=[]
                for o_key in frame_dict[id_int].keys():
                    new[0,:]=frame_dict[id_int][o_key][2:6]
                    cls_conf_other.append(frame_dict[id_int][o_key][6])
                    bbox_xywh_other=np.vstack((bbox_xywh_other,new.copy()))
                cls_conf_other=np.array(cls_conf_other)
                # cls_conf_other=np.ones(len(cls_conf_other))
                bbox_xywh_other[:,0]=bbox_xywh_other[:,0]+bbox_xywh_other[:,2]/2
                bbox_xywh_other[:,1]=bbox_xywh_other[:,1]+bbox_xywh_other[:,3]/2

            bbox_xywh_ensemble=None
            if bbox_xywh is None and bbox_xywh_other is not None:
                bbox_xywh_ensemble=bbox_xywh_other
                cls_conf_ensemle=cls_conf_other
            


            if bbox_xywh_ensemble is not None:

                # do tracking
                outputs = self.deepsort.update(bbox_xywh_ensemble, cls_conf_ensemle, im)
            
                new=np.zeros((len(outputs),8))
                # draw boxes for visualization
                if len(outputs) > 0:
                    new[:,-2]=1
                    new[:,1]=outputs[:,-1]
                    new[:,2:4]=outputs[:,:2]
                    new[:,4:6]=outputs[:,2:4]-outputs[:,:2]
                    if idx_frame==3:
                        new[:,0]=1
                        M=np.vstack((M,new.copy()))
                        new[:,0]=2
                        M=np.vstack((M,new.copy()))

                    new[:,0]=idx_frame
                    M=np.vstack((M,new))

                    bbox_xyxy = outputs[:,:4]
                    identities = outputs[:,-1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                elif idx_frame>3:
                    print('================================lost_frame===================================')

            end = time.time()
            print("id_video:{} frame:{} time: {:.03f}s, fps: {:.03f}".format(id_video,idx_frame,end-start, 1/(end-start)))

            # if self.args.display:
            #     cv2.imshow("test", ori_im)
            #     cv2.waitKey(1)
            

            if self.args.save_path:        
                video.write(ori_im)        #把图片写进视频
        M=M[1:].astype(int)
        np.savetxt(txt_name, M ,fmt='%i', delimiter=",")
        video.release() #释放


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--VIDEO_PATH", type=str,default='/home/liujierui/proj/Dataset/A-data')
    parser.add_argument("--VIDEO_PATH", type=str,default='/home/liujierui/proj/Dataset/B-data')
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=False)
    parser.add_argument("--frame_interval", type=int, default=1)

    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/A-track/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    
    # label='c2_101_rec_mmdet_81.5_3_score0.7_iout0.55_B'
    # label='c2_B_rec_74.8_4_score0.7_iout0.55_B'
    label='c2_B_rec_76.7_6_score0.7_iout0.55_B'
    # label='after_rec_dif_i0cut_c2_B_rec_74.8_3_score0.7_iout0.55_B_75.592_ite'
    # label='cascade_rcnn_score0.5_iout0.55_B'
    # label='c2_rec_mmdet_81.9_4_score0.8_iout0.55_B'
    args.save_path=args.save_path+label

    with VideoTracker(cfg, args) as vdo_trk:
        # vdo_trk.run(1)
        # vdo_trk.run(4)
        # vdo_trk.run(5)
        # vdo_trk.run(9)
        # vdo_trk.run(10)
        vdo_trk.run(2)
        vdo_trk.run(3)
        vdo_trk.run(6)
        vdo_trk.run(8)
        vdo_trk.run(11)
        vdo_trk.run(12)
