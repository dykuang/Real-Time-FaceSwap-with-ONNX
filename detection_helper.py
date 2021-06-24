# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 22:20:24 2020

@author: dykua

helpler functions for onnx face detection
"""
import numpy as np
import cv2

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    #print(boxes)
    #print(confidences)
    
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        #print(confidences.shape[1])
        probs = confidences[:, class_index]
        #print(probs)
        mask = probs > prob_threshold
        probs = probs[mask]
        
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        #print(subset_boxes)
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

class BBox(object):
    # bbox is a list of [left, right, top, bottom]
    def __init__(self, bbox):
        self.left = bbox[0]
        self.right = bbox[1]
        self.top = bbox[2]
        self.bottom = bbox[3]
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    # scale to [0,1]
    def projectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape))     
        for i, point in enumerate(landmark):
            landmark_[i] = ((point[0]-self.x)/self.w, (point[1]-self.y)/self.h)
        return landmark_

    # landmark of (5L, 2L) from [0,1] to real range
    def reprojectLandmark(self, landmark):
        landmark_= np.asarray(np.zeros(landmark.shape)) 
        for i, point in enumerate(landmark):
            x = point[0] * self.w + self.x
            y = point[1] * self.h + self.y
            landmark_[i] = (x, y)
        return landmark_
    
object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652]])
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape, img):
    h, w, _ = img.shape
    K = [w, 0.0, w // 2,
         0.0, w, h // 2,
         0.0, 0.0, 1.0]
    D = [0, 0, 0.0, 0.0, 0]
    cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
    dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35]])
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    return reprojectdst, euler_angle

# def drawLandmark(img, bbox, landmark):
#     '''
#     Input:
#     - img: gray or RGB
#     - bbox: type of BBox
#     - landmark: reproject landmark of (5L, 2L)
#     Output:
#     - img marked with landmark and bbox
#     '''
#     img_ = img.copy()
#     cv2.rectangle(img_, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
#     for x, y in landmark:
#         cv2.circle(img_, (int(x), int(y)), 3, (0,255,0), -1)
#     return img_

# def drawLandmark_multiple(img, bbox, landmark):
#     '''
#     Input:
#     - img: gray or RGB
#     - bbox: type of BBox
#     - landmark: reproject landmark of (5L, 2L)
#     Output:
#     - img marked with landmark and bbox
#     '''
#     cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
#     for x, y in landmark:
#         cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
#     return img

# def drawLandmark_Attribute(img, bbox, landmark,gender,age):
#     '''
#     Input:
#     - img: gray or RGB
#     - bbox: type of BBox
#     - landmark: reproject landmark of (5L, 2L)
#     Output:
#     - img marked with landmark and bbox
#     '''
#     cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
#     for x, y in landmark:
#         cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)
#         if gender.argmax()==0:
#                 # -1->female, 1->male; -1->old, 1->young
#                 cv2.putText(img, 'female', (int(bbox.left), int(bbox.top)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
#         else:
#                 cv2.putText(img, 'male', (int(bbox.left), int(bbox.top)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),3)
#         if age.argmax()==0:
#                 cv2.putText(img, 'old', (int(bbox.right), int(bbox.bottom)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 3)
#         else:
#                 cv2.putText(img, 'young', (int(bbox.right), int(bbox.bottom)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 3)
#     return img


# def drawLandmark_only(img, landmark):
#     '''
#     Input:
#     - img: gray or RGB
#     - bbox: type of BBox
#     - landmark: reproject landmark of (5L, 2L)
#     Output:
#     - img marked with landmark and bbox
#     '''
#     img_=img.copy()
#     #cv2.rectangle(img_, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
#     for x, y in landmark:
#         cv2.circle(img_, (int(x), int(y)), 3, (0,255,0), -1)
#     return img_
