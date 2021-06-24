# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:05:02 2021

@author: dykua

face-swap with webcam test
"""
import cv2
from detection import Face_Detector, Landmark_Detector
import numpy as np

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def get_triangle_from_index(img, landmarks, index):
    tr_pt1 = landmarks[index[0]]
    tr_pt2 = landmarks[index[1]]
    tr_pt3 = landmarks[index[2]]
    triangle = np.array([tr_pt1, tr_pt2, tr_pt3], np.int32)
    rect = cv2.boundingRect(triangle)
    (x, y, w, h) = rect
    cropped_triangle = img[y: y + h, x: x + w]
    cropped_tr_mask = np.zeros((h, w), np.uint8)
    points = np.array([[tr_pt1[0] - x, tr_pt1[1] - y],
                      [tr_pt2[0] - x, tr_pt2[1] - y],
                      [tr_pt3[0] - x, tr_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_tr_mask, points, 255)
    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle,
                                        mask=cropped_tr_mask)
    
    return cropped_triangle, points, rect, cropped_tr_mask

class face_swap():
    def __init__(self, from_face, face_detector= Face_Detector(), 
                 lmk_detector = Landmark_Detector(),
                 out_size = (400, 300)):
        
       self.out_size = out_size 
        
       img = cv2.imread(from_face)
       # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
       img = cv2.resize(img, self.out_size)
       self.mask = np.zeros(img.shape[:-1]) 
        
       self.face_detector =  face_detector
       self.lmk_detector = lmk_detector
       
       face_boxes, _ = self.face_detector.detect(img) 
       self.face_box = face_boxes[0].astype(np.int32) # get the first 

       self.lmk_pts, PRY_3d = self.lmk_detector.detect(img, self.face_box)  # get landmarks
       # self.lmk_pts = self.lmk_pts.astype(np.int32)
       
       self.lmk_pts = [tuple(pt) for pt in self.lmk_pts]

       points = np.array(self.lmk_pts, np.int32)
       convexhull = cv2.convexHull(points)
        #cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
       cv2.fillConvexPoly(self.mask, convexhull, 255)
    
       # Delaunay triangulation
       rect = cv2.boundingRect(convexhull)
       subdiv = cv2.Subdiv2D(rect)
       for pt in self.lmk_pts:
            subdiv.insert(pt)  # insert list of tuples
       triangles = subdiv.getTriangleList()
       triangles = np.array(triangles, dtype=np.int32)
    
       # get the Landmark points indexes of each triangle
       self.indexes_triangles = []
       for t in triangles:
           pt1 = (t[0], t[1])
           pt2 = (t[2], t[3])
           pt3 = (t[4], t[5])
           index_pt1 = np.where((points == pt1).all(axis=1))
           index_pt1 = extract_index_nparray(index_pt1)
           index_pt2 = np.where((points == pt2).all(axis=1))
           index_pt2 = extract_index_nparray(index_pt2)
           index_pt3 = np.where((points == pt3).all(axis=1))
           index_pt3 = extract_index_nparray(index_pt3)
           if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
               triangle = [index_pt1, index_pt2, index_pt3]
               self.indexes_triangles.append(triangle)
               
       self.cropped_triangles = []        
       for triangle_index in self.indexes_triangles:
            cropped_triangle, tri_pts, _, _= get_triangle_from_index(img, self.lmk_pts, triangle_index)
            self.cropped_triangles.append([cropped_triangle,tri_pts])
                
    def run(self, to_face, moving_lmks):
        self.new_face = np.zeros_like(to_face) 
        moving_lmks = [tuple(pt) for pt in moving_lmks]
        points = np.array(moving_lmks, np.int32) # array of tuples
        convexhull = cv2.convexHull(points)
        
        for triangle_index, cropped_triangle in zip(self.indexes_triangles, self.cropped_triangles):
            moving_patch, moving_pts, moving_rect, moving_mask = get_triangle_from_index(to_face, moving_lmks, triangle_index)
            
            # Warp triangles
            (x, y, w, h) = moving_rect
            fixed_pts = np.float32(cropped_triangle[1])
            moving_pts = np.float32(moving_pts)
            M = cv2.getAffineTransform(fixed_pts, moving_pts)
            warped_triangle = cv2.warpAffine(cropped_triangle[0], M, (w, h), flags=cv2.WARP_FILL_OUTLIERS, # this did the trick
                                             borderMode=cv2.BORDER_DEFAULT) #https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=moving_mask)
            
            # Reconstructing destination face
            new_face_rect_area = self.new_face[y: y + h, x: x + w]
            new_face_rect_area_gray = cv2.cvtColor(new_face_rect_area, cv2.COLOR_BGR2GRAY)
            _, mask_triangles_designed = cv2.threshold(new_face_rect_area_gray, 4, 255, 
                                                       cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
            # mask_triangles_designed = cv2.adaptiveThreshold(new_face_rect_area_gray,255,
            #                                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                                    cv2.THRESH_BINARY,11,2)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
            # mask_triangles_designed = cv2.morphologyEx(mask_triangles_designed, cv2.MORPH_CLOSE, kernel)
            
            warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
            # warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=moving_mask)

            new_face_rect_area = cv2.add(new_face_rect_area, warped_triangle)
            self.new_face[y: y + h, x: x + w] = new_face_rect_area
            # self.new_face[y: y + h, x: x + w] = cv2.GaussianBlur(new_face_rect_area, (5,5), 0)
            
        new_face_mask = np.zeros_like(to_face[...,0])
        new_head_mask = cv2.fillConvexPoly(new_face_mask, convexhull, 255)
        new_face_mask = cv2.bitwise_not(new_head_mask)
        new_head_noface = cv2.bitwise_and(to_face, to_face, mask=new_face_mask)
        
        rough = cv2.add(new_head_noface, self.new_face)
        (x, y, w, h) = cv2.boundingRect(convexhull)
        center_face = (int((x + x + w) / 2), int((y + y + h) / 2))
        fine = cv2.seamlessClone(rough, to_face, new_head_mask, center_face, cv2.NORMAL_CLONE)
        return fine
        
        
        
        
            
if __name__ == '__main__':
    import os
    portraits = os.listdir('portraits/')
    num_faces = len(portraits)
    current_face = np.random.randint(num_faces)
    FS = face_swap( os.path.join('portraits', portraits[current_face % num_faces]) )
    face_detector = Face_Detector()
    lmk_detector = Landmark_Detector()
    
    
    # # check image
    # img = cv2.imread("target.png")
    # img = cv2.resize(img, (400, 300))
    
    # face_boxes, _ = face_detector.detect(img)
    # face_box = face_boxes[0] # get the first 
    # face_box = face_box.astype(np.int32)
    # landmarks_points, PRY_3d = lmk_detector.detect(img, face_box)  # get landmarks
    # landmarks_points = landmarks_points.astype(np.int32)
    
    # frame = FS.run(img, landmarks_points)
    # cv2.imwrite('swap.png', frame)
    
    
    # check cam
    feed = cv2.VideoCapture(0)
    while True:
        ret, frame = feed.read() 
        frame = cv2.resize(frame, (400,300))
        bboxes, _ = face_detector.detect(frame)  # get faces
        if len(bboxes) != 0:
            bbox = bboxes[0] # get the first 
            bbox = bbox.astype(np.int)
            lmks, PRY_3d = lmk_detector.detect(frame, bbox)  # get landmarks
            lmks = lmks.astype(np.int)
            frame = FS.run(frame,lmks)
            cv2.imshow("Face Swap", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("n"):
            current_face += 1
            try:
                FS = face_swap( os.path.join('portraits', portraits[current_face % num_faces]) )
            except:
                print('Swapping failed for {}'.format(portraits[current_face % num_faces]))
        
        elif cv2.waitKey(1) & 0xFF == ord("p"):
            current_face -= 1
            try:
                FS = face_swap( os.path.join('portraits', portraits[current_face % num_faces]) )
            except:
                print('Swapping failed for {}'.format(portraits[current_face % num_faces]))
        
        elif cv2.waitKey(1) & 0xFF == ord("q"):
            break
            