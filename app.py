from flask import Flask, render_template,request, Response
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app=Flask(__name__)



face_classifier = cv2.CascadeClassifier('./Harcascade/haarcascade_frontalface_default.xml')
classifier=load_model('model.h5')
class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']



def detect():
    depression = 0
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
                depression=depression - 3    

            if depression>=40:
                label="Depression"

                
            cv2.putText(img, label, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            # cv2.imshow("Depression Detection",img)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def vid_detect(file_path):
    print("success")
    depression = 0
    cap = cv2.VideoCapture(file_path)
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


            if depression>=50:
                label="Depression"

                
            cv2.putText(img, label, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            # cv2.imshow("Depression Detection",img)
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



@app.route('/')
def home():
    return render_template("home.html")

@app.route('/Start_detection')
def cam_start():
    return render_template("start.html")

@app.route('/cam')
def cam_detect():
    return render_template( "cam detect.html")

@app.route('/video_feed')
def video_feed():
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def upload():
    return render_template('video start.html')

fil=[]

@app.route('/upload', methods=["POST"])
def vid():
    file=request.files["uploads"]
    basepath = os.path.dirname(__file__)
    # print(basepath)
    global file_path
    file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
    file.save(file_path)
    print(file_path)
    fil.append(file_path)
    print(fil[0])
    return render_template("video detect.html")


@app.route('/upload_feed')
def upload_feed():
    video=fil[0]
    print(video)
    return Response(vid_detect(video), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)    