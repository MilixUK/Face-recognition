# This module of the application is responsible for collection the photo samples of the users face
# and generating a document required by recogniser.

#Importing required packages.
import cv2
import os
import numpy as np
from PIL import Image
# Creating a video stram and window to display it.
cam = cv2.VideoCapture(0)
cam.set(3, 640) # video width
cam.set(4, 480) # video height
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Path to a folder which stores phot samples
path = 'dataset'
# Creating a recogniser
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Creating a detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
# For each person, enter one numeric face id

# The below function is used in the process of tcollectiong the photo samples.
def photoSampleCollection(cam, face_detector):
    # Collecting the user Id whioch is used to differentiate different users of the system.
    face_id = input('\n Please input user id number (intiger) and press Enter >>>  ')

    print("\n [INFO] Face capture in progress. Look directly into camera lens and wait ...")
    # Face photo capture process in progress.
    count = 0
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            # Saving captured images into the datasets folder.
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff # Press 'ESC' to exit.
        if k == 27:
            break
        elif count >= 50: # Take 50 face sample and the video stream.
             break
    # Close the window and video stream.
    print("\n [INFO]  Face capture completed.")
    cam.release()
    cv2.destroyAllWindows()
# The below function is responsible for training the software to recognise faces using recogniser and detector.
def faceTraining(path, detector, recognizer):
        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L') # Converting to grayscale.
                img_numpy = np.array(PIL_img,'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
            return faceSamples,ids
        print ("\n [INFO] Training faces in progress. It will take several seconds. Plese wait ...")
        faces,ids = getImagesAndLabels(path)
        # Training the recogniser.
        recognizer.train(faces, np.array(ids))
        # Saveing the yml model into initiation/initiation.yml
        recognizer.write('initiation/initiation.yml')
        # Printing total number of faces trained and program termination.
        print("\n [INFO] {0} faces has been trained so far. Initiation process compleated.".format(len(np.unique(ids))))


    
photoSampleCollection(cam, face_detector)
faceTraining(path,detector , recognizer)
