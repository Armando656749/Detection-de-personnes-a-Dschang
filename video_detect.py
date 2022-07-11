import numpy as np
import cv2
import glob
import math
from object_detection import ObjectDetection



# Modele yolo v4
net = cv2.dnn.readNet("dnn_model/yolov4.weights", "dnn_model/yolov4.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), # plus grand est l'image, meilleur est la precision mais le processus est lent
                     scale=1/255)
# Liste des differentes categories d'objets que le modele de reseau (yolo) detecte pour Coco
classes = []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
print(classes)

# Initialisation du detecteur d'objet
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")

track_id = 0
tracking_objects = {}
center_points_prev_frame = []
count=0
while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    print("Trame N°: {}".format(count))
    # Enregistre le centre de gravite des objets detectes pour une trame donnees
    center_points_cur_frame = []

    # detection d'objet sur les trames
    (class_ids, scores, bboxes) = od.detect(frame)
    for box in bboxes:
        (x,y,w,h) = box
        cx = int(x + w/2)
        cy = int(y + h/2)
        center_points_cur_frame.append((cx,cy))
        print("Trame N°{} boxes: ".format(count), (x,y,w,h))
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        #cv2.circle(frame, (cx,cy), 3, (0,0,255), -1)

    #print("Centre d'objets de la trame actuelle:")
    #print(center_points_cur_frame)
    #print("Centre d'objets de la trame precedente:")
    #print(center_points_prev_frame)

    # Tracking d'objets
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1]) #hypot calcul la distrance euclidienne
                if distance <20:
                    tracking_objects[track_id] = pt
                    track_id += 1
                
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        for object_id, pt2 in tracking_objects_copy.items():

            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0]-pt[0], pt2[1]-pt[1])
                # on met a jour la position des objets
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue
            # on retire son identifiant s'il n'existe pas
            if not object_exists:
                tracking_objects.pop(object_id)

        # On ajoute une nouvelle etiquette sur l'objet detecte
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1


    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 3, (0,0,255),-1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1]- 7), 0, 1, (0,0,255), 2)

    print("tracking_objects")
    print(tracking_objects)
    cv2.imshow("video",frame)

    # ici je copies le centre de la trame actuelle
    center_points_prev_frame = center_points_cur_frame.copy()
    key = cv2.waitKey(0)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()