# ZTE_challenge_sort_top9
About 2020 ZTE_challenge_multi-target tracking - top9 Vereary队

代码结构

.
├── detect_with_mmdet
│   ├── afterprocess_json.py
│   ├── configs
│   ├── demo
│   ├── docker
│   ├── docs
│   ├── mmdet
│   ├── pretrainmodels
│   ├── pytest.ini
│   ├── README.md
│   ├── setup.py
│   ├── test_Jr3.sh
│   ├── tests
│   ├── tools
│   ├── train_Jr3_101.sh
│   └── train_Jr3.sh
├── stage1_fairmot
│   ├── assets
│   ├── experiments
│   ├── outputs
│   ├── src
│   └── videos
├── stage2_deepsort
│   ├── after_process_B.py
│   ├── after_process.py
│   ├── clean_insert_clean_frame.py
│   ├── clean_insert_frame.py
│   ├── configs
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
│   ├── yolov3_deepsort_mycrop_ensembledet_more.py
│   ├── yolov3_deepsort_mycrop_ensembledet.py
│   ├── yolov3_deepsort_mycrop_ite.py
│   └── yolov3_deepsort.py
├── to_coco.py
└── to_coco_train.py

部分代码参考了[mmdet](https://github.com/open-mmlab/mmdetection)、[fairmot](https://github.com/ifzhang/FairMOT)、[deepsort](https://github.com/ZQPei/deep_sort_pytorch)
