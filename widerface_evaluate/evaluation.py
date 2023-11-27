
import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed
import cv2
import torch
import math


saveResultImg = False
output_folder = '/home/yanan/cv/yolov5-face/img/'

# NME, metric for keypoint detection
def mean_euclidean_distance(pred_points, true_points):
    distances = np.linalg.norm(pred_points - true_points, axis=1)
    mean_distance = np.mean(distances)
    return mean_distance

# MAE, metric for alignment
def euler_angles_mae(pred_angles, true_angles):
    # Convert angles to radians
    pred_angles_rad = np.radians(pred_angles)
    true_angles_rad = np.radians(true_angles)

    # Calculate absolute differences in radians
    abs_diff_rad = np.abs(pred_angles_rad - true_angles_rad)

    # Convert absolute differences back to degrees
    abs_diff_deg = np.degrees(abs_diff_rad)

    # Calculate the mean absolute error
    mae = np.mean(abs_diff_deg)
    
    return mae

# Accuracy, metric for face detection
def calculate_iou(gt, pred):
    box1 = [int(pred[0]), int(pred[1]), int(pred[0]+pred[2]), int(pred[1]+pred[3])]
    box2 = [int(gt[0]-gt[2]/2), int(gt[1]-gt[3]/2), int(gt[0]+gt[2]/2), int(gt[1]+gt[3]/2)]
    # Calculate the intersection coordinates
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou

def face_detection_accuracy(true_boxes, pred,gt, img, iou_threshold=0.5):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    pred_boxes = pred[...,0:4]
    landmark_pred = []
    landmark_gt = []
    alignment_pred = []
    alignment_gt = []
    for i in range(len(true_boxes)):
        box_matched = False
        for j in range(len(pred_boxes)):
            iou = calculate_iou(true_boxes[i], pred_boxes[j])
            if iou >= iou_threshold:
                true_positive += 1
                box_matched = True
                if saveResultImg:
                    if len(pred[j]) > 15:
                        angle = -pred[j][15]
                        center = (int(pred[j][9]), int(pred[j][10]))
                        roll = int(pred[j][15])
                        pitch = int(pred[j][16])
                        yaw = int(pred[j][17])

                        roll = roll * np.pi / 180
                        pitch = pitch * np.pi / 180
                        yaw = -(yaw * np.pi / 180)
                        size = 30
                        (tdx, tdy) = center
                        x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
                        y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

                        x2 = -size * (-np.cos(yaw) * np.sin(roll)) + tdx
                        y2 = -size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

                        x3 = size * (np.sin(yaw)) + tdx
                        y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy
                        cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (221,234,17), int(img.shape[0] * 0.005))
                        cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0,255,0), int(img.shape[0] * 0.005))
                        cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (0,0,255), int(img.shape[0] * 0.005))

                        cv2.putText(img, ('pitch: {}').format(int(pitch)), (int(pred[j][0]-50),int(pred[j][1]+pred[j][3]+80)), cv2.FONT_HERSHEY_SIMPLEX, 1, (221,234,17), thickness=2, lineType=2)
                        cv2.putText(img, ('roll: {}').format(int(roll)), (int(pred[j][0]-50),int(pred[j][1]+pred[j][3]+120)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2, lineType=2)
                        cv2.putText(img, ('yaw: {}').format(int(yaw)), (int(pred[j][0]-50),int(pred[j][1]+pred[j][3]+160)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=2, lineType=2)
                    else:
                        angle = 1
                    rect = ((pred_boxes[j][0]+pred_boxes[j][2]//2, pred_boxes[j][1]+pred_boxes[j][3]//2), (pred_boxes[j][2], pred_boxes[j][3]), angle)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                    for k in range(5,15,2):
                        cv2.circle(img, (int(pred[j][k]), int(pred[j][k+1])), 1, (0,255,255),4)
                if gt[i, 5:15].sum() == -10:
                    break

                landmark_pred.append(pred[j, 5:15])
                landmark_gt.append(gt[i,5:15].numpy())
                if len(pred[j]) > 15:
                    alignment_pred.append(pred[j,15:18])
                    alignment_gt.append(gt[i,15:18].numpy())
                break

        if not box_matched:
            false_negative += 1

    false_positive = len(pred_boxes) - true_positive

    accuracy = true_positive / (true_positive + false_positive + false_negative) if (true_positive + false_positive + false_negative) > 0 else 0.0

    return accuracy,landmark_pred,landmark_gt,alignment_pred,alignment_gt, img
# mAP for face detection
def calculate_precision_recall(gt_boxes, pred_boxes, confidence_scores, iou_threshold=0.5):
    """
    Calculate precision and recall values for different confidence score thresholds.
    """
    num_gt_boxes = len(gt_boxes)
    num_pred_boxes = len(pred_boxes)

    sorted_indices = np.argsort(confidence_scores)[::-1]
    tp = np.zeros(num_pred_boxes)
    fp = np.zeros(num_pred_boxes)
    precision = np.zeros(num_pred_boxes)
    recall = np.zeros(num_pred_boxes)

    for i in range(num_pred_boxes):
        pred_box = pred_boxes[sorted_indices[i]]
        max_iou = 0
        max_iou_index = -1

        for j in range(num_gt_boxes):
            iou = calculate_iou(gt_boxes[j], pred_box)
            if iou > max_iou:
                max_iou = iou
                max_iou_index = j

        if max_iou >= iou_threshold:
            if not gt_boxes[max_iou_index][-1]:  # Check if the ground truth box is not already matched
                tp[i] = 1
                gt_boxes[max_iou_index][-1] = 1  # Mark the ground truth box as matched
            else:
                fp[i] = 1
        else:
            fp[i] = 1

        precision[i] = np.sum(tp) / (np.sum(tp) + np.sum(fp))
        recall[i] = np.sum(tp) / num_gt_boxes

    return precision, recall

def calculate_ap(precision, recall):
    """
    Calculate Average Precision (AP) from precision and recall values.
    """
    ap = 0
    recall_range = np.arange(0.0, 1.1, 0.1)

    for r in recall_range:
        recall_mask = (recall >= r)
        if np.any(recall_mask):
            ap += np.max(precision[recall_mask])

    ap /= 11.0  # 11 recall values (0.0, 0.1, ..., 1.0)
    return ap


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] == '':
            continue
        # a = float(line[4])
        line = list(map(float, line))
        boxes.append(line)
    boxes = np.array(boxes)
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes

def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, 4])
            _max = np.max(v[:, 4])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, 4] = (v[:, 4] - min_score)/diff


def evaluation(pred, gt_path, baseline,threshold, iou_thresh=0.5):
    pred = get_preds(pred)
    norm_score(pred)
    accuracy_total = []
    nme_total = []
    mae_total = []
    ap_total = []
    for folder, k in pred.items():
        for pic, v in k.items():
            with open(gt_path+pic+'_refine.txt', 'r') as f:
                if len(v) == 0:
                    accuracy_total.append(0)
                    continue
                lines = f.readlines()
                gt = []
                img0 = cv2.imread(gt_path+pic+'.jpg')
                h,w, c = img0.shape
                for line in lines:
                    line = line.replace('  ', ' ')
                    line = line.rstrip('\r\n').split(' ')
                    line = list(map(float, line))
                    line[1:5:2] = [i * w for i in line[1:5:2]]
                    line[2:5:2] = [i * h for i in line[2:5:2]]
                    line[5:15:2] = [i * w for i in line[5:15:2]]
                    line[6:15:2] = [i * h for i in line[6:15:2]]
                    line[17] = line[22]
                    line[18:22:2] = [i * w for i in line[18:22:2]]
                    line[19:22:2] = [i * h for i in line[19:22:2]]
                    gt.append(line)
            gt = torch.as_tensor(gt)
            v = v[v[...,4]>threshold]
            pred_boxes = v[...,0:4]
            if baseline:
                gt_boxes = gt[...,1:5].numpy()
            else:
                gt_boxes = gt[...,18:22].numpy()

            accuracy, landmark_pred,landmark_gt,alignment_pred,alignment_gt, img = face_detection_accuracy(gt_boxes,v, gt, img0)
            accuracy_total.append(accuracy)
            
            gt_boxes = [[x1, y1, w1, h1, 0] for x1, y1, w1, h1 in gt_boxes]  # 0 at the end is a flag for matching
            pred_boxes = [[x2, y2, w2, h2] for x2, y2, w2, h2 in pred_boxes]
            confidence_scores = v[...,4]

            precision, recall = calculate_precision_recall(gt_boxes, pred_boxes, confidence_scores)
            ap = calculate_ap(precision, recall)
            ap_total.append(ap)

            
            if len(landmark_pred)>0:
                nme = mean_euclidean_distance(np.array(landmark_pred), np.array(landmark_gt))
                nme_total.append(nme)
            if len(alignment_pred)>0:
                mae = euler_angles_mae(np.array(alignment_pred), np.array(alignment_gt))
                mae_total.append(mae)
            if saveResultImg:
                cv2.imwrite(output_folder+pic+'.jpg', img)
            # print(folder, pic)

    if len(accuracy_total)>0:
        accuracy_final = np.array(accuracy_total).mean()
        map_final = np.array(ap_total).mean()
    else:
        accuracy_final = 0
        map_final = 0
    if len(nme_total)>0:
        nme_final = np.array(nme_total).mean()
    else:
        nme_final = 'none'
    if len(mae_total)>0:
        mae_final = np.array(mae_total).mean()
    else:
        mae_final = 'none'
    print(args.pred)
    print("==================== Results ====================")
    print("Accuracy: {}".format(accuracy_final))
    print("mAP: {}".format(map_final))
    print("NME: {}".format(nme_final))
    print("MAE: {}".format(mae_final))
    print("=================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="widerface_evaluate/widerface_txt_origin_newdata/")
    parser.add_argument('-g', '--gt', default='dataset/widerface_multitask/val/')
    parser.add_argument('-b', '--baseline', default=False)
    parser.add_argument('-th', '--threshold', default=0.1)
    args = parser.parse_args()
    evaluation(args.pred, args.gt, args.baseline, args.threshold)












