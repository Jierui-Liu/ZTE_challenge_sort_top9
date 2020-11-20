# ZTE_challenge_sort_top9

About 2020 ZTE_challenge_multi-target tracking - top9 Vereary队

# 0. 环境&依赖

**环境：**

+ Ubuntu 16.04
+ CUDA 10.1 CUDNN 7.6.4
+ 1080Ti
+ 内存128G

# 1. 代码结构
本项目部分代码参考了[mmdet](https://github.com/open-mmlab/mmdetection)、[fairmot](https://github.com/ifzhang/FairMOT)、[deepsort](https://github.com/ZQPei/deep_sort_pytorch)

其中detect_with_mmdet、stage1_fairmot、stage2_deepsort的配置分别参照[mmdet](https://github.com/open-mmlab/mmdetection)、[fairmot](https://github.com/ifzhang/FairMOT)、[deepsort](https://github.com/ZQPei/deep_sort_pytorch)（可兼容）

```
.
├── detect_with_mmdet
│   ├── afterprocess_json.py  //将检测结果转换为跟踪代码的读取格式      
│   ├── configs
│   ├── demo
│   ├── docker
│   ├── docs
│   ├── mmdet
│   ├── pretrainmodels
│   ├── pytest.ini
│   ├── README.md
│   ├── setup.py
│   ├── test_Jr3.sh //检测器推理脚本
│   ├── tests
│   ├── tools
│   ├── train_Jr3_101.sh //检测器训练脚本 
│   └── train_Jr3.sh //检测器训练脚本
├── stage1_fairmot
│   ├── assets
│   ├── experiments
│   ├── outputs
│   ├── src //第一阶段跟踪执行代码 
│   └── videos
├── stage2_deepsort
│   ├── after_process_B.py //后处理代码v4 
│   ├── after_process.py //后处理代码v3 
│   ├── clean_insert_clean_frame.py //后处理代码v2 
│   ├── clean_insert_frame.py //后处理代码v1 
│   ├── configs
│   ├── deep_sort
│   ├── demo
│   ├── detector
│   ├── insert_frame.py
│   ├── insert_visual.py
│   ├── README.md
│   ├── rec_frame.py
│   ├── to_SST.py
│   ├── transform
│   ├── utils
│   ├── yolov3_deepsort_mycrop_ensembledet_more.py //deepsort迭代式跟踪代码v3 
│   ├── yolov3_deepsort_mycrop_ensembledet.py //deepsort迭代式跟踪代码v2
│   ├── yolov3_deepsort_mycrop_ite.py //deepsort迭代式跟踪代码v1
│   └── yolov3_deepsort.py //deepsort初始跟踪代码 
├── to_coco.py
└── to_coco_train.py
```

# 2. 后处理算法

定义了修正框和在时间轴上膨胀腐蚀的后处理（after_process.py），在得到跟踪结果）之后，对于空缺帧数低于45且空缺前后检测框iou较大的轨迹进行插帧，然后再利用不同框的重叠关系有选择的抑制。.后处理之后分数能涨1%~2%。将所有框的高比宽约束在3.5~1之间，然后对跟踪结果进行时间轴上膨胀腐蚀的后处理即可得到最终的输出结果。


