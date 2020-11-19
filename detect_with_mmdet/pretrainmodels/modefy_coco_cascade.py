
def main():
    #gen coco pretrained weight
    import torch
    root='/home/liujierui/proj/mmdetection/pretrainmodels/'
    num_classes = 2
    model_coco = torch.load(root+"cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth") # weight
    # model_coco = torch.load("cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166 (1).pth") # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] =\
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] =\
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] =\
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][ :num_classes, :]
    # bias 
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] =\
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] =\
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] =\
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][ :num_classes]
    # save new model
    torch.save(model_coco, root+"cascade_rcnn_r50_fpn_20e_coco_pretrained_weights_classes_%d.pth" % num_classes)
    # torch.save(model_coco, "cascade_rcnn_dconv_c3-c5_r50_fpn_1x_coco_pretrained_weights_classes_%d.pth" % num_classes)
if __name__ == "__main__":
    main()