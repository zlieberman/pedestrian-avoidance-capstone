from .YOLOv3_archive import YOLOv3_archive


__all__ = ['build_YOLOv3_archive']

def build_YOLOv3_archive(cfg, use_cuda):
    return YOLOv3_archive(cfg.YOLOV3.CFG, cfg.YOLOV3.WEIGHT, cfg.YOLOV3.CLASS_NAMES, 
                    score_thresh=cfg.YOLOV3.SCORE_THRESH, nms_thresh=cfg.YOLOV3.NMS_THRESH, 
                    is_xywh=True, use_cuda=use_cuda)
