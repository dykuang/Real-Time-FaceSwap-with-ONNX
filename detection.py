# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:23:38 2021

@author: dykua
"""
# import time
import cv2
import numpy as np
import onnxruntime as ort
# import os
from detection_helper import predict, get_head_pose
from scipy.special import softmax
from tracker import Tracker


def crop_image(orig, bbox):
    bbox = bbox.copy()
    image = orig.copy()
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    face_width = (1 + 2 * 0.2) * bbox_width
    face_height = (1 + 2 * 0.2) * bbox_height
    center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
    bbox[0] = max(0, center[0] - face_width // 2)
    bbox[1] = max(0, center[1] - face_height // 2)
    bbox[2] = min(image.shape[1], center[0] + face_width // 2)
    bbox[3] = min(image.shape[0], center[1] + face_height // 2)
    bbox = bbox.astype(np.int)
    crop_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    h, w, _ = crop_image.shape
    
    return crop_image, ([h, w, bbox[1], bbox[0]])

def reshape_for_polyline(array):
    """Reshape image so that it works with polyline."""
    return np.array(array, np.int32).reshape((-1, 1, 2))
    
def face_sketch(landmarks, color = (255, 255, 255), thickness=3):
    black_image = np.zeros(frame.shape, np.uint8)
    
    jaw = reshape_for_polyline(landmarks[0:17])
    left_eyebrow = reshape_for_polyline(landmarks[22:27])
    right_eyebrow = reshape_for_polyline(landmarks[17:22])
    nose_bridge = reshape_for_polyline(landmarks[27:31])
    lower_nose = reshape_for_polyline(landmarks[30:35])
    left_eye = reshape_for_polyline(landmarks[42:48])
    right_eye = reshape_for_polyline(landmarks[36:42])
    outer_lip = reshape_for_polyline(landmarks[48:60])
    inner_lip = reshape_for_polyline(landmarks[60:68])
    
    cv2.polylines(black_image, [jaw], False, color, thickness)
    cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [nose_bridge], False, color, thickness)
    cv2.polylines(black_image, [lower_nose], True, color, thickness)
    cv2.polylines(black_image, [left_eye], True, color, thickness)
    cv2.polylines(black_image, [right_eye], True, color, thickness)
    cv2.polylines(black_image, [outer_lip], True, color, thickness)
    cv2.polylines(black_image, [inner_lip], True, color, thickness)

    return black_image

class Face_Detector:
    def __init__(self, input_size = (320, 240), model_path = "./models/version-RFB-320.onnx"):
        # model_path = "./models/version-RFB-320.onnx"
        self.sess = ort.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.input_size = input_size

    def detect(self, orig_image):
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1]) # channel first
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        # time_time = time.time()
        confidences, boxes = self.sess.run(None, {self.input_name: image})
        # print("Face Detector inference time:{}".format(time.time() - time_time))
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, 0.8)
        return boxes, probs
    
class Landmark_Detector:
    def __init__(self, detection_size=(160, 160), model_path = "./models/slim_160_latest.onnx"):
        self.sess = ort.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.detection_size = detection_size
        self.tracker = Tracker()

    def detect(self, img, bbox):
        # crop_image, detail = self.crop_image(img, bbox)
        cropped_image, detail = crop_image(img, bbox)
        cropped_image = cv2.resize(cropped_image, self.detection_size)
        cropped_image = (cropped_image - 127.0) / 127.0
        cropped_image = np.array([np.transpose(cropped_image, (2, 0, 1))]).astype(np.float32)
        # start = time.time()
        raw = self.sess.run(None, {self.input_name: cropped_image})[0][0]
        # end = time.time()
        # print("ONNX Inference Time: {:.6f}".format(end - start))
        landmark = raw[0:136].reshape((-1, 2))
        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3]
        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2]
        landmark = self.tracker.track(img, landmark)
        _, PRY_3d = get_head_pose(landmark, img)
        return landmark, PRY_3d[:, 0]
        # return landmark, None
    
           

if __name__ == '__main__':  
    from faceswap_cam import face_swap
    FS = face_swap("portraits/elsa.jpg")
    face_detector = Face_Detector()
    lmk_detector = Landmark_Detector()

    cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (frame.shape[1], frame.shape[0]))
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        bboxes, _ = face_detector.detect(frame)  # get faces
        if len(bboxes) != 0:
            bbox = bboxes[0] # get the first 
            bbox = bbox.astype(np.int)
            lmks, PRY_3d = lmk_detector.detect(frame, bbox)  # get landmarks
            lmks = lmks.astype(np.int)
            
            frame = FS.run(frame,lmks) # special effects
            
            # sketch = face_sketch(lmks)
            # frame = cv2.bitwise_or(frame, frame, sketch)
            frame = cv2.rectangle(frame, tuple(bbox[0:2]), tuple(bbox[2:4]), (0, 0, 255), 1, 1)

            
            # for point in lmks:
            #     frame = cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1, 1)
            # frame = cv2.putText(frame, "Pitch: {:.4f}".format(PRY_3d[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #                     (0, 255, 0), 1, 1)
            # frame = cv2.putText(frame, "Yaw: {:.4f}".format(PRY_3d[1]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #                     (0, 255, 0), 1, 1)
            # frame = cv2.putText(frame, "Roll: {:.4f}".format(PRY_3d[2]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            #                     (0, 255, 0), 1, 1)
        cv2.imshow("Landmark Detection", frame)
        if cv2.waitKey(27) == ord("q"):
            break
        # out.write(frame)
    
    # out.release()
    cap.release()
    cv2.destroyAllWindows() 
