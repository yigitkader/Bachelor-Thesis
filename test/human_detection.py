###
#Copyright by YigitKader
###

import cv2
import numpy as np



#take frame from video
camera = cv2.VideoCapture(0)

#face_cascade=cv2.CascadeClassifier('cascades/haarcascade_profile.xml')
#frontal_face_cascade=cv2.CascadeClassifier('cascades/frontal_face.xml')
frontal_face_extended=cv2.CascadeClassifier('src/frontal_face_extended.xml')

while (1):
    ret,frame=camera.read()

    grey_ton = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Convertcolor(cvtColor)
    faces = frontal_face_extended.detectMultiScale(grey_ton,1.1,2)

    for(x,y,w,h) in faces:
        #show frame
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3) #framei göster,sol üst ,sağ üst koordinatla,renk,kalınlık


    cv2.imshow('original',frame)

    if cv2.waitKey(25) & 0xFF ==ord('q'):
        break


camera.release()
cv2.destroyAllWindows()
