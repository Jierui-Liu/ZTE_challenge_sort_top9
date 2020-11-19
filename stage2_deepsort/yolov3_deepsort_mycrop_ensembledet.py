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
        
def get_area_iou_rate(box_xywh1,box_xywh2):
    b1=box_xywh1.copy()
    b1[2:]=box_xywh1[:2]+box_xywh1[2:]
    b2=box_xywh2.copy()
    b2[2:]=box_xywh2[:2]+box_xywh2[2:]
    delta_h = min(b1[2], b2[2]) - max(b1[0], b2[0])
    delta_w = min(b1[3], b2[3])-max(b1[1], b2[1])
    if delta_h < 0 or delta_w < 0:
        return 0
    else:
        overlap = 1.0*delta_h * delta_w
        area1=box_xywh1[2]*box_xywh1[3]
        rate=overlap/area1
        return rate



def crop(img):
    # 默认 width>height bias[i]=[xi,yi]
    height,width,_=img.shape
    ratio=int(round(width/height))
    if ratio<=1:
        imgs=[img]
        bias=[[0,0]]
        return imgs,bias
    else:
        pass
        num=ratio+1
        step=(width-height)/ratio
        bias=[[step*i,0] for i in range(num)]+[[0,0]]
        imgs=[img[:,int(step*i):int(step*i+height),:] for i in range(num)]+[img]
        return imgs,bias


def detect_imgs(detector,imgs,bias):
    dets_all=np.zeros((0,5))
    for i in range(len(imgs)):
        bbox_xywh, cls_conf, cls_ids = detector(imgs[i])
        if bbox_xywh is None:
            continue
        
        mask = cls_ids==0

        bbox_xywh = bbox_xywh[mask]
        # bbox_xywh[:,3:] *= 1.2 # bbox dilation just in case bbox too small
        cls_conf = cls_conf[mask]
        
        dets=np.zeros((len(cls_conf),5))
        dets[:,:4]=bbox_xywh
        dets[:,4]=cls_conf
        if dets is not None:
            dets[:,0]=dets[:,0]+bias[i][0]
            dets[:,1]=dets[:,1]+bias[i][1]
            dets_all=np.vstack((dets_all,dets))
    if len(dets_all)==0:
        return None,None,None
    dets_xyxy_c=dets_all.copy()
    dets_xyxy_c[:,0]=dets_xyxy_c[:,0]-dets_xyxy_c[:,2]/2
    dets_xyxy_c[:,1]=dets_xyxy_c[:,1]-dets_xyxy_c[:,3]/2
    dets_xyxy_c[:,2:4]=dets_all[:,2:4]+dets_all[:,:2]

    # indexs=py_nms(dets_xyxy_c,0.3)
    
    dets_xyxy_c=torch.FloatTensor(dets_xyxy_c).cuda()
    indexs=boxes_nms(boxes=dets_xyxy_c[:,:4],scores=dets_xyxy_c[:,4],nms_thresh=0.3)
    indexs=indexs.detach().cpu().numpy()

    return dets_all[indexs,:4],dets_all[indexs,4],np.zeros(len(indexs))

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
        path=join(self.args.VIDEO_PATH,'Track'+str(id_video),'img1')#把img去掉
        filelist_0 = os.listdir(path) #获取该目录下的所有文件名
        filelist=[str(i) for i in range(1,len(filelist_0)+1)]   
        avi_name = join(self.args.save_path,str(id_video) + ".avi")#导出路径

        dirtxt_read='/home/liujierui/proj/FairMOT-master/outputs/A-track_all_hrnet_v2_w18_score0.3_nms0.3_conf0.35'
        # dirtxt_read='/home/liujierui/proj/FairMOT-master/outputs/A-track_all_dla34_score0.3_nms0.3_conf0.45'
        # dirtxt_read='/home/liujierui/proj/centerNet-deep-sort-master/demo/A-track/cn_crop_score0.7_ensemble_hrnet_fair0.4'
        # dirtxt_read='/home/liujierui/proj/centerNet-deep-sort-master/demo/A-track/cn_score0.4_ensemble_hrnet_fair0.35'
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

            # do detection
            imgs,bias=crop(im)
            bbox_xywh, cls_conf, cls_ids=detect_imgs(self.detector,imgs,bias)
            # bbox_xywh, cls_conf, cls_ids = self.detector(im)

            
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids==0

                bbox_xywh = bbox_xywh[mask]
                # bbox_xywh[:,3:] *= 1.2 ##调整 # bbox dilation just in case bbox too small
                bbox_xywh[:,3:] *= 1.2 ##调整 # bbox dilation just in case bbox too small
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

            if bbox_xywh is not None and bbox_xywh_other is not None:
                bbox_xywh_temp=bbox_xywh.copy()
                bbox_xywh_other_temp=bbox_xywh_other.copy()
                bbox_xywh_temp[:,0]=bbox_xywh[:,0]-bbox_xywh[:,2]/2
                bbox_xywh_temp[:,1]=bbox_xywh[:,1]-bbox_xywh[:,3]/2
                bbox_xywh_other_temp[:,0]=bbox_xywh_other[:,0]-bbox_xywh_other[:,2]/2
                bbox_xywh_other_temp[:,1]=bbox_xywh_other[:,1]-bbox_xywh_other[:,3]/2

                bbox_xywh_ensemble=bbox_xywh_other
                cls_conf_ensemle=cls_conf_other
                for i,bbox_xywh0 in enumerate(bbox_xywh_temp):
                    add=True
                    for bbox_xywh0_other in bbox_xywh_other_temp:
                        iou=get_iou(bbox_xywh0,bbox_xywh0_other)
                        rate=get_area_iou_rate(bbox_xywh0,bbox_xywh0_other)
                        if iou>0.5 or rate>0.8:
                        # if iou>0.3 or rate>0.8:
                            add=False
                            break
                    if add:
                        bbox_xywh_ensemble=np.vstack((bbox_xywh[i][np.newaxis,:],bbox_xywh_ensemble))
                        cls_conf_ensemle=np.hstack((cls_conf[i],cls_conf_ensemle))
                        
            elif bbox_xywh is None and bbox_xywh_other is not None:
                bbox_xywh_ensemble=bbox_xywh_other
                cls_conf_ensemle=cls_conf_other
            elif bbox_xywh is not None and bbox_xywh_other is None:
                bbox_xywh_ensemble=bbox_xywh
                cls_conf_ensemle=cls_conf
            else:
                bbox_xywh_ensemble=None


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
    parser.add_argument("--VIDEO_PATH", type=str,default='/home/liujierui/proj/Dataset/A-data')
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
    
    # label='init_score0.8_nms0.3_0.85_crop_ensemble_hrnet_fair0.35_cn0.4'
    label='init_score0.8_nms0.3_0.85_crop_ensemble_hrnet_fair0.35_iou0.5'
    # label='init_score0.8_nms0.3_0.85_crop_ensemble_hrnet_fair0.35_ioudist0.9'
    # label='init_score0.8_nms0.3_0.85_crop_ensemble_fair0.45_cn0.6'
    args.save_path=args.save_path+label

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run(1)
        vdo_trk.run(4)
        vdo_trk.run(5)
        vdo_trk.run(9)
        vdo_trk.run(10)
