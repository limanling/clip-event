import math
import numpy as np

# croped = img[cy-r:cy+r, cx-r:cx+r]

# def normalize_bbox(bbox, width, height):
#     bbox = bbox.copy()
#     bbox[0] /= width
#     bbox[1] /= height
#     bbox[2] /= width
#     bbox[3] /= height
#     return bbox
def normalize_bbox(bbox, width, height):
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max = x_min / width, x_max / width
    y_min, y_max = y_min / height, y_max / height
    return (x_min, y_min, x_max, y_max)

def normalize_bbox_batch(bbox, width, height):
    bbox = bbox.copy()
    bbox[:, 0] /= width
    bbox[:, 1] /= height
    bbox[:, 2] /= width
    bbox[:, 3] /= height
    return bbox 


def patch_from_norm_bbox(bbox_norm, patch_size=7):
    x_min, y_min, x_max, y_max = bbox_norm
    x_min_idx, y_min_idx, x_max_idx, y_max_idx = x_min * patch_size, y_min * patch_size, x_max * patch_size, y_max * patch_size
    x_min_idx, y_min_idx, x_max_idx, y_max_idx = math.floor(x_min_idx), math.floor(y_min_idx), math.ceil(x_max_idx), math.ceil(y_max_idx)
    return (x_min_idx, y_min_idx, x_max_idx, y_max_idx)

def patch_from_norm_bbox_batch(bbox_norm, patch_size=7):
    bbox = bbox_norm.copy()
    bbox[:, 0] = math.floor(bbox[:, 0] * patch_size)
    bbox[:, 1] = math.floor(bbox[:, 1] * patch_size)
    bbox[:, 2] = math.ceil(bbox[:, 2] * patch_size)
    bbox[:, 3] = math.ceil(bbox[:, 2] * patch_size)
    return bbox 

def IoU(boxA, boxB):
    # order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def isCorrect(bbox_annot, bbox_pred, iou_thr=.5):
    iou_value_max = 0.0
    for bbox_p in bbox_pred:
        for bbox_a in bbox_annot:
            iou_value = IoU(bbox_p, bbox_a)
            iou_value_max = max(iou_value, iou_value_max)
            if iou_value >= iou_thr:
                return 1, iou_value
    return 0, iou_value_max

def union(bbox):
    if len(bbox) == 0:
        return []
    if type(bbox[0]) == type(0.0) or type(bbox[0]) == type(0):
        bbox = [bbox]
    maxes = np.max(bbox, axis=0)
    mins = np.min(bbox, axis=0)
    return [mins[0], mins[1], maxes[2], maxes[3]]