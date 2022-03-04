def xywh_to_tlwh(bbox_xywh):
    x,y,w,h = bbox_xywh
    tl_x = max(int(x-w/2), 0)
    tl_y = max(int(y-h/2), 0)
    return tl_x,tl_y,w,h


def tlwh_to_xywh(bbox_tlwh):
    x,y,w,h = bbox_tlwh
    c_x = int((x+w)/2)
    c_y = int((y+h)/2)
    return c_x,c_y,w,h


def xywh_to_xyxy(bbox_xywh, width, height):
    x,y,w,h = bbox_xywh
    x1 = max(int(x-w/2), 0)
    x2 = min(int(x+w/2), width-1)
    y1 = max(int(y-h/2), 0)
    y2 = min(int(y+h/2), height-1)
    return x1,y1,x2,y2


def xyxy_to_xywh(bbox_xyxy):
    x1,y1,x2,y2 = bbox_xyxy
    c_x = int((x1+x2)/2)
    c_y = int((y1+y2)/2)
    w = int(x2-x1)
    h = int(y2-y1)
    return c_x,c_y,w,h


def tlwh_to_xyxy(bbox_tlwh, width, height):
    x,y,w,h = bbox_tlwh
    x1 = max(int(x),0)
    x2 = min(int(x+w),self.width-1)
    y1 = max(int(y),0)
    y2 = min(int(y+h),self.height-1)
    return x1,y1,x2,y2


def xyxy_to_tlwh(bbox_xyxy):
    x1,y1,x2,y2 = bbox_xyxy
    c_x = int(x1)
    c_y = int(y1)
    w = int(x2-x1)
    h = int(y2-y1)
    return c_x,c_y,w,h