import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('./Harcascade/haarcascade_frontalface_default.xml')
classifier=load_model('model.h5')
class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

depression = 0

def detect():
    cap = cv2.VideoCapture(0)
    while True:
        ret,img = cap.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        allfaces = []
        rects = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            allfaces.append(roi_gray)
            rects.append((x, w, y, h))
        i = 0
        for face in allfaces:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (rects[i][0] + int((rects[i][1]/2)),
                        abs(rects[i][2] - 10))
            i = + 1

            if depression<=-30:
                depression= depression + 27
        
            if label=="Sad":
                depression=depression + 1
                print("depress ",depression)               
            else:
                depression=depression - 1    

            if depression>=40:
                label="Depression"   
                
            cv2.putText(img, label, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            cv2.imshow("Depression Detection",img)
            
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

