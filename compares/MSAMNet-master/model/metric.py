import  numpy as np
import torch.nn as nn
import torch
from skimage import measure
import  numpy

class PRMetric():
    def __init__(self, dis_eps):
        super(PRMetric, self).__init__()
        self.tp = 0
        self.fp = 0
        self.pred = 0
        self.gt = 0
        self.dis_thresh = dis_eps

    def update(self, preds, labels):
        preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)           
        tp = 0
        fp = 0
        gt_matched = np.zeros(len(labels))

        for i, pred in enumerate(preds):
            if len(labels) > 0:
                distances = [np.linalg.norm(np.array(pred["center"])- np.array(gt["center"])) for gt in labels]
                min_dis = min(distances)
                if min_dis <= self.dis_thresh:
                    min_idx = distances.index(min_dis)
                    if not gt_matched[min_idx]:
                        tp += 1
                        gt_matched[min_idx] = 1
                    else:
                        fp += 1
                else:
                    fp += 1
            else:
                fp += 1
        self.tp   += tp
        self.fp   += fp
        self.pred  += len(preds)
        self.gt += len(labels)

    def get(self):
        recall      = self.tp / (self.gt   + 0.001)
        precision   = self.tp / (self.pred + 0.001)

        pd_rate = recall
        fa_rate = 1-precision

        return recall, precision, pd_rate, fa_rate

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.pred = 0
        self.gt = 0

class APMetric():
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(APMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros((self.nclass,self.bins))
        self.fp_arr = np.zeros((self.nclass,self.bins))
        self.pred_arr = np.zeros((self.nclass,self.bins))
        self.gt_arr=np.zeros((self.nclass,self.bins))
        self.global_pred_dict = {}
        self.global_gt_dict = {}
        self.image_id = 0
        # self.reset()

    def update(self, preds, labels):
        for pred in preds:
            class_id = pred["class_id"]
            if class_id not in self.global_pred_dict:
                self.global_pred_dict[class_id] = []
            self.global_pred_dict[class_id].append({
                "image_id": self.image_id,
                "confidence": pred["confidence"],
                "center": pred["center"],
                "size": pred["size"]
            })

        for gt in labels:
            class_id = gt["class_id"]
            if class_id not in self.global_gt_dict:
                self.global_gt_dict[class_id] = []
            self.global_gt_dict[class_id].append({
                "image_id": self.image_id,
                "center": gt["center"],
                "size": gt["size"]
            })

        self.image_id += 1

    def get(self):
        aps = []
        for class_id in self.global_pred_dict.keys():
            pred_boxes = self.global_pred_dict.get(class_id, [])
            gt_boxes = self.global_gt_dict.get(class_id, [])
            if len(gt_boxes) == 0:
                continue
            ap = calculate_ap(pred_boxes, gt_boxes)
            aps.append(ap)
        return np.mean(aps)
        
    def reset(self):
        self.tp_arr = np.zeros((self.nclass,self.bins))
        self.fp_arr = np.zeros((self.nclass,self.bins))
        self.pred_arr = np.zeros((self.nclass,self.bins))
        self.gt_arr = np.zeros((self.nclass,self.bins))
        self.global_pred_dict = {}
        self.global_gt_dict = {}
        self.image_id = 0

def calculate_ap(pred_boxes, gt_boxes, distance_threshold=3):
    pred_boxes = sorted(pred_boxes, key=lambda x: x["confidence"], reverse=True)

    tp = np.zeros(len(pred_boxes))  # True Positive
    fp = np.zeros(len(pred_boxes))  # False Positive
    matched_gt = {}  # 存储每张图像已匹配的真实框索引

    for i, pred in enumerate(pred_boxes):
        image_id = pred["image_id"]
        pred_center = pred["center"]
        min_distance = float('inf')
        match_idx = -1

        gt_boxes_image = [gt for gt in gt_boxes if gt["image_id"] == image_id]
        for j, gt in enumerate(gt_boxes_image):
            if image_id in matched_gt and j in matched_gt[image_id]:
                continue 
            gt_center = gt["center"]
            distance = np.linalg.norm(np.array(pred_center)-np.array(gt_center))
            if distance < min_distance and distance < distance_threshold:
                min_distance = distance
                match_idx = j

        if match_idx != -1:
            tp[i] = 1
            matched_gt.setdefault(image_id, set()).add(match_idx)
        else:
            fp[i] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / len(gt_boxes)

    ap = np.trapz(precision, recall)
    return ap

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled, predicted = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_predict += predicted
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixRecall = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        pixPrecision = 1.0 * self.total_correct / (np.spacing(1) + self.total_predict)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixRecall, pixPrecision, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_predict = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()
    pixel_predicted = predict.sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled, pixel_predicted


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union

