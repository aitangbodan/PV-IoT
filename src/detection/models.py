import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_detector(model_name, num_classes, pretrained=True):
    if model_name == 'faster_rcnn':
        # 使用Faster R-CNN with ResNet-50 FPN
        model = detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        # 替换分类头
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    elif model_name == 'yolo':
        # 简单返回None，实际需要额外安装库，这里为示例
        raise NotImplementedError("YOLO not implemented in this code")
    elif model_name == 'retinanet':
        model = detection.retinanet_resnet50_fpn(pretrained=pretrained)
        in_features = model.head.classification_head.cls_score.in_features
        model.head.classification_head.cls_score = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")