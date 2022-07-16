import streamlit as st
import cv2
import os, shutil
import numpy as np
from PIL import Image
import os
import cv2
import mediapipe as mp
from angle_calc import angle_calc
import mimetypes
import imutils
from tensorflow.keras.models import load_model
from time import sleep
import time
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import pandas as pd
import numpy as np
import altair as alt
import math as m
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

count=0
score=0
thicc=2
rpred=[99]
lpred=[99]
timestamp_list, angry_list, scared_list, happy_list, sad_list, neutral_list, surprised_list, neck_inc_list, torso_inc_list, sleep_drow_list=0,0,0,0,0,0,0,0,0,0
analysis_dictt=0
drow_score_list=0
def facetrainfun():
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids


    face_id = st.text_input('Enter Your Wellness APP ID: ', "")

    col1, col2 = st.columns(2)

    dataset_path="dataset"
    trainer_path="trainer"


    col1.write("Original Frame")
    col2.write("Captured Frame")

    camera=col1.empty()
    captured_camera=col2.empty()

    finalrecog=st.empty()

    framecount=st.empty()

    if st.button('Start Capture'):
        try:
            face_id=int(face_id)
            # num_list = [ord(x) - 96 for x in face_id]
            # num_list=''.join(num_list)
            # face_id=int(num_list)
            if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
                shutil.rmtree(dataset_path)

            if os.path.exists(trainer_path) and os.path.isdir(trainer_path):
                shutil.rmtree(trainer_path)

            os.mkdir(dataset_path)
            os.mkdir(trainer_path)
            cam = cv2.VideoCapture(0)
            cam.set(3, 640) # set video width
            cam.set(4, 480) # set video height

            face_detector = cv2.CascadeClassifier('facerecog/haarcascade_frontalface_default.xml')

            st.write("[INFO] Initializing face capture. Look the camera and wait ...")
            # Initialize individual sampling face count
            count = 0

            while(True):

                ret, img = cam.read()
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                camera.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                for (x,y,w,h) in faces:

                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                    count += 1
                    framecount.write("Number of Frames Recieved: "+str(count))
                    # Save the captured image into the datasets folder
                    captured_camera.image(gray[y:y+h,x:x+w])
                    cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    

                    # cv2.imshow('image', img)

                k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= 80: # Take 80 face sample and stop video
                    break

            # Do a bit of cleanup
            st.write("[INFO] Exiting Program and cleanup stuff")
            captured_camera.write("")
            camera.write("")
            cam.release()
            cv2.destroyAllWindows()

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier("facerecog/haarcascade_frontalface_default.xml");


            st.write("[INFO] Training faces. It will take a few seconds. Wait ...")
            faces,ids = getImagesAndLabels(dataset_path)
            recognizer.train(faces, np.array(ids))

            # Save the model into trainer/trainer.yml
            recognizer.write(trainer_path+'/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

            # Print the numer of faces trained and end program
            st.write("[INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
            os.rmdir("dataset")
        except Exception as e:
            st.error("Please Enter Valid Wellness ID "+ str(e))


def mainappfun():
    class_labels = ['angry', 'scared', 'happy', 'sad', 'surprised',
                'neutral']

    COLORS = {
        'angry': (0, 0, 255),
        'scared': (0, 128, 255),
        'happy': (0, 255, 255),
        'sad': (255, 0, 0),
        'surprised': (178, 255, 102),
        'neutral': (160, 160, 160)
    }

    face = cv2.CascadeClassifier('drowss\haar cascade files\haarcascade_frontalface_alt.xml')
    leye = cv2.CascadeClassifier('drowss\haar cascade files\haarcascade_lefteye_2splits.xml')
    reye = cv2.CascadeClassifier('drowss\haar cascade files\haarcascade_righteye_2splits.xml')

    trainer_path='trainer'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path+'/trainer.yml')
    cascadePath = "facerecog/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)


    lbl=['Close','Open']

    model = load_model('drowss/models/cnncat2.h5')
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL


    face_classifier=cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    classifier = load_model('model-ep061-loss0.795-val_loss0.882.h5')


    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    mimetypes.init()

    cascade_file = 'resources/haarcascade_frontalface_default.xml'
    det = cv2.CascadeClassifier(cascade_file)

    # Colors.
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)

    # Calculate angle.
    def findAngle(x1, y1, x2, y2):
        theta = m.acos( (y2 -y1)*(-y1) / (m.sqrt(
            (x2 - x1)**2 + (y2 - y1)**2 ) * y1) )
        degree = int(180/m.pi)*theta
        return degree

    def posture_fun(img):
        # success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        pose1=[]
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z=[]
                h, w,c = img.shape
                x_y_z.append(lm.x)
                x_y_z.append(lm.y)
                x_y_z.append(lm.z)
                x_y_z.append(lm.visibility)
                pose1.append(x_y_z)
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id%2==0:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
                else:
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)

            # Use lm and lmPose as representative of the following methods.
            lm = results.pose_landmarks
            lmPose  = mpPose.PoseLandmark
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

            # Right shoulder.
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
            angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))
            print(angle_text_string)
    #############NEWC0DE
        rula,reba=angle_calc(pose1)
        print(rula,reba)
        if (rula != "NULL") and (reba != "NULL"):
            if int(rula)>3:
                print("Rapid Upper Limb Assessment Score : "+rula+" Posture not proper in upper body")
                print("Posture not proper in upper body","Warning")
            else:
                print("Rapid Upper Limb Assessment Score : "+rula)
            if int(reba)>4:
                print("Rapid Entire Body Score : "+reba+" Posture not proper in your body")
                print("Posture not proper in your body","Warning")
            else:
                print("Rapid Entire Body Score : "+reba)
        else:
            print("Posture Incorrect")
        
        img = imutils.resize(img, width=380)
        return img, neck_inclination, torso_inclination

    def emotions_fun(frame, face_classifier, classifier):
        # ret,frame=cap.read()
        labels=[]
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi=roi_gray.astype('float')/255.0
                roi=img_to_array(roi)
                roi=np.expand_dims(roi,axis=0)

                preds=classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position=(x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

                emotion_list=[]
                prob_list=[]
                datadictt={}
                for i, (emotion, probability) in enumerate(zip(class_labels, preds)):
                    datadictt[emotion]=probability
                    emotion_list.append(emotion)
                    prob_list.append(probability)
                # probability=[i/sum(probability) for i in probability]
                chart_data =pd.DataFrame({'index':emotion_list, 'Values': prob_list})
                data = pd.melt(chart_data.reset_index(), id_vars=["index"])
                data=data[data['variable']=="Values"]


            else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        img = imutils.resize(frame, width=380)
        return img, data, datadictt

    def drowsiness(frame):

        global count
        global score
        global thicc
        global rpred
        global lpred

        # ret, frame = cap.read()
        height,width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
        left_eye = leye.detectMultiScale(gray)
        right_eye = reye.detectMultiScale(gray)

        cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye= r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = model.predict_classes(r_eye)
            if(rpred[0]==1):
                lbl='Open'
            if(rpred[0]==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = model.predict_classes(l_eye)
            if(lpred[0]==1):
                lbl='Open'
            if(lpred[0]==0):
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        # if(rpred[0]==1 or lpred[0]==1):
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        if(score<0):
            score=0
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>15):
            #person is feeling sleepy so we beep the alarm
            # cv2.imwrite(os.path.join(path,'image.jpg'),frame)
            print("FEELIN SLEEPY")
            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)
        img = imutils.resize(frame, width=380)
        return img, count

    # st.set_page_config(layout="wide")
    chkbox=st.empty()
    agree = chkbox.checkbox('Enable Face Tracking')
    col1, col2, col3 = st.columns(3)
    img1emp, img2emp, img3emp= col1.empty(), col2.empty(), col3.empty()
    FRAME_WINDOW1 = img1emp.image([])
    FRAME_WINDOW2 = img2emp.image([])
    FRAME_WINDOW3 = img3emp.image([])

    pl = col2.empty()
    drow_chart = col3.empty()
    # neck_gauge=col1.empty()
    # torso_gauge=col1.empty()
    neck_torso_plot=col1.empty()

    return_vals=[0,0,0,0]
    cap = cv2.VideoCapture(0)
    # drlist=[0,0,2,[99],[99]]

    minW = 0.1*cap.get(3)
    minH = 0.1*cap.get(4)
    global drow_score_list
    drow_score_list=[]

    global timestamp_list, angry_list, scared_list, happy_list, sad_list, neutral_list, surprised_list, neck_inc_list, torso_inc_list, sleep_drow_list
    global analysis_dictt

    timestamp_list=[]
    angry_list=[]
    scared_list=[]
    happy_list=[]
    sad_list=[]
    neutral_list=[]
    surprised_list=[]
    neck_inc_list=[]
    torso_inc_list=[]
    sleep_drow_list=[]

    while (1):
        time.sleep(0.1)
        success, img = cap.read()
        if(agree):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
            )
            
            
            for(x,y,w,h) in faces:

                # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

                id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                # Check if confidence is less them 100 ==> "0" is perfect match 
                if (confidence < 100):
                    # id = names[id]
                    id='Face'
                    confidence = "  {0}%".format(round(100 - confidence))
                    xx, yy, ww, hh=x, y, w, h
                else:
                    id = 'unknown'
                    confidence = "  {0}%".format(round(100 - confidence))

            # img1, img2, img3= img.copy(), img.copy(), img.copy()
            img1, img2, img3 = (img.copy()), img[yy:y+hh, xx:xx+ww].copy(), img[yy:y+hh, xx:xx+ww].copy()
        else:
            img1, img2, img3= img.copy(), img.copy(), img.copy()
        try:
            pos, neck_inclination, torso_inclination=posture_fun(cv2.flip(img1, 1))
            try:
                emp, data, datadictt=emotions_fun(cv2.flip(img2, 1), face_classifier, classifier)
            except:
                datadictt={i:0 for i in class_labels}
                # ret,frame=cap.read()
                emp = imutils.resize(cv2.flip(img2, 1), width=380)
                chart_data =pd.DataFrame({'index':class_labels, 'Values': [5 for i in class_labels]})
                data = pd.melt(chart_data.reset_index(), id_vars=["index"])
                data=data[data['variable']=="Values"]
            drow, drow_intensity=drowsiness(cv2.flip(img3, 1))
            # cv2.imshow("Image",  np.hstack([pos, emp, drow]))
            # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW1.image(cv2.cvtColor(pos, cv2.COLOR_BGR2RGB))
            FRAME_WINDOW2.image(cv2.cvtColor(emp, cv2.COLOR_BGR2RGB),  width=pos.shape[1])
            FRAME_WINDOW3.image(cv2.cvtColor(drow, cv2.COLOR_BGR2RGB),  width=pos.shape[1])
        
            drow_score_list.append(score)

            if(len(drow_score_list)>10):
                drow_score_list=drow_score_list[-10:]
            
            chart_data = pd.DataFrame(
            drow_score_list,
            columns=['drowsiness score'])

            drow_chart.line_chart(chart_data)

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    x=alt.X("value", type="quantitative", title=""),
                    y=alt.Y("index", type="nominal", title=""),
                    color=alt.Color("index", type="nominal", title=""),
                    order=alt.Order("value", sort="descending"),
                ).properties(
                    width=800,
                    height=300)
            )

            pl.altair_chart(chart, use_container_width=True)

            fig = make_subplots(rows=2, cols=1,
            specs=[[{'type' : 'domain'}],[{'type' : 'domain'}]],)

            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = 60-int(neck_inclination),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Neck Inclination"},
                gauge = {
                    'shape': "angular",
                    'axis': {'range': [0, 60]}}), row=1, col=1)

            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = int(torso_inclination),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Torso Inclination"},
                gauge = {
                    'shape': "angular",
                    'axis': {'range': [0, 18]}}), row=2, col=1)

            neck_torso_plot.plotly_chart(fig, use_container_width=True)

            try:
                print('appending')
                timenow=time.time()
                ang=datadictt['angry']
                sca=datadictt['scared']
                sadd=datadictt['sad']
                surp=datadictt['surprised']
                neut=datadictt['neutral']
                happ=datadictt['happy']
                

                timestamp_list.append(timenow)
                angry_list.append(ang)
                scared_list.append(sca)
                happy_list.append(happ)
                sad_list.append(sadd)
                neutral_list.append(neut)
                surprised_list.append(surp)

                neck_inc_list.append(60-int(neck_inclination))
                torso_inc_list.append(int(torso_inclination))
                sleep_drow_list.append(score)
                analysis_dictt={'timedataa':timestamp_list,'angry':angry_list,'scared':scared_list,
                'happy':happy_list,'sad':sad_list,'neutral':neutral_list,'surprised':surprised_list,
                'neck_inclination':neck_inc_list,'torso_inclination':torso_inc_list,'drowsiness_score':sleep_drow_list}
                analysis_dictt=str(analysis_dictt).replace("'","\"")
                f = open("analysisdatatemp", "w")
                f.write(analysis_dictt)
                f.close()
            except Exception as e:
                print(e)
                pass
        except Exception as e:
            print(e)
            pass
            # break
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
