import torch
def main():
    #gen coco pretrained weight
    root='/home/liujierui/proj/mmdetection/pretrainmodels/'
    import torch
    num_classes = 2
    # model_coco = torch.load(root+"faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth") # weight
    model_coco = torch.load(root+"faster_rcnn_r101_fpn_1x_20181129-d1468807.pth") # weight
    model_coco["state_dict"]["bbox_head.fc_cls.weight"] =model_coco["state_dict"]["bbox_head.fc_cls.weight"][ :num_classes, :]
    # bias model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] =model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.fc_cls.bias"] =model_coco["state_dict"]["bbox_head.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.fc_reg.weight"] =model_coco["state_dict"]["bbox_head.fc_reg.weight"][ :num_classes*4, :]
    # bias model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] =model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.fc_reg.bias"] =model_coco["state_dict"]["bbox_head.fc_reg.bias"][ :num_classes*4]
    # save new model
    # torch.save(model_coco, root+"faster_rcnn_r50_fpn_1x_coco_pretrained_weights_classes_%d.pth" % num_classes)
    torch.save(model_coco, root+"faster_rcnn_r101_fpn_1x_coco_pretrained_weights_classes_%d.pth" % num_classes)
if __name__ == "__main__":
    main()