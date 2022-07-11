import numpy as np
import cv2
import glob



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

# Je testes le detecteur yolo sur une image avant de generaliser
img = cv2.imread("images_test/trame_196.jpg")

class_ids, scores, bboxes =model.detect(img, nmsThreshold=0.4, confThreshold=0.5)
print("")
print("class_ids:",class_ids)

tracking = {}
obj_id = 0
for class_id, score, bboxe in zip(class_ids, scores, bboxes):
    x, y, w, h = bboxe
    score = np.round(score,2)
    if class_id == 0:
        class_name = classes[class_id]
        cx = int(x + w/2)
        cy = int(y + h/2)
        tracking[obj_id] = (cx,cy)
        obj_id +=1
    cv2.putText(img, class_name + str(obj_id) + ": " +str(score), (x-3,y), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)
    cv2.circle(img, (cx,cy), 3, (0,0,255), -1)
print("objets detecter")
print(tracking)
cv2.imshow("image", img)
cv2.waitKey(0)

