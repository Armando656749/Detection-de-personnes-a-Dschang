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

cap = cv2.VideoCapture(0)
count=0

while True:
    ret, img = cap.read()
    if ret is False:
        break
    count+=1

    class_ids, scores, bboxes =model.detect(img, nmsThreshold=0.4, confThreshold=0.5)
    cpt = 0
    tracks = {}
    for class_id, score, bboxe in zip(class_ids, scores, bboxes):
        score = round(score,2)
        x, y, w, h = bboxe
        if class_id == 0:
            cpt += 1
            class_name = classes[class_id]
            cx = int(x + w/2)
            cy = int(y + h/2)
            tracks[cpt] = (cx, cy)
            cv2.putText(img, class_name+str(cpt)+": "+str(score), (x-3,y), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)
        cv2.circle(img, (cx,cy), 3, (0,0,255), -1)
    print("Nombre de personnes detecter:", cpt)

    # Relativement a la variable cpt, on doit faire un test: cpt>seuil alors print (surchage)
    # A defaut print(pas de surcharge)
    
    # Enregistrement des trames de la camera
    #if count <= 20:
    #    cv2.imwrite("camera_{}.jpg".format(count), img)
    
    print("")
    print(tracks)
    cv2.imshow("camera", img)
    key = cv2.waitKey(0)

    if key == "q":
        break
cap.release()
cv2.destroyAllWindows()