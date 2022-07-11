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

# ici je recuperes le chemin d'acces des images contenu dans le dossier images_test
images_path = glob.glob("images_test/*.jpg")

for path in images_path:

    img = cv2.imread(path)
    region_1 = [(39,525),(227,557),(336,263),(265,263)]
    region_2 = [(628,544),(794,540),(563,235),(478,243)]
    region_1_ids = set()
    region_2_ids = set()

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
            in_region_1 = cv2.pointPolygonTest(np.array(region_1),(cx,cy), False)
            in_region_2 = cv2.pointPolygonTest(np.array(region_2),(cx,cy), False)
            if in_region_1 > 0:
                region_1_ids.add(obj_id)
            elif in_region_2 > 0:
                region_2_ids.add(obj_id)
            else:
                continue
        
        cv2.putText(img, class_name + str(obj_id) + ": " +str(score), (x-3,y), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)
        cv2.circle(img, (cx,cy), 3, (0,0,255), -1)
        # regions
        cv2.polylines(img, [np.array(region_1)], True, (0,255,255),4)
        cv2.polylines(img, [np.array(region_2)], True, (0,255,255),4)

    print("objets detecter")
    print(tracking)
    print("")
    print("len region_1_ids:", region_1_ids)
    print("len region_2_ids:", region_2_ids)
    cv2.imshow("image", img)
    cv2.waitKey(0)
# Appliquons le detecteur sur les images
'''
for path in images_path:
    img = cv2.imread(path)
    class_ids, scores, bboxes =model.detect(img, nmsThreshold=0.4, confThreshold=0.5)
    cpt = 0
    tracks = {}
    for class_id, score, bboxe in zip(class_ids, scores, bboxes):
        x, y, w, h = bboxe
        if class_id == 0:
            cpt += 1
            class_name = classes[class_id]
            cx = int(x + w/2)
            cy = int(y + h/2)
            tracks[cpt] = (cx, cy)
            cv2.putText(img, class_name+str(cpt), (x-3,y), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 1)
        cv2.circle(img, (cx,cy), 3, (0,0,255), -1)
    print("Nombre de personnes detecter:", cpt)

    # Relativement a la variable cpt, on doit faire un test: cpt>seuil alors print (surchage)
    # A defaut print(pas de surcharge)

    print("")
    print(tracks)
    cv2.imshow("image", img)
    cv2.waitKey(0)
'''