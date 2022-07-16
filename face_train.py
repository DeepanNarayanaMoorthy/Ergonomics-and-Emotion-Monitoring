import streamlit as st
import cv2
import os, shutil
import numpy as np
from PIL import Image
import os


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
    except Exception as e:
        st.error("Please Enter Valid Wellness ID "+ str(e))

