

#Copyright by YigitKader

# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognition
from statistics import mode
from zdatasets.datasets import get_labels
from datasets.inference import draw_bounding_box
from datasets.inference import draw_text
from datasets.preprocessor import preprocess_input
from datasets.inference import apply_offsets




USE_WEBCAM = True # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './src/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []



cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./test/testvdo.mp4') # Video file source




#take frame from camera
camera = cv2.VideoCapture(0)


frontal_face_extended=cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')
# frontal_face_extended=cv2.CascadeClassifier('cascades/frontal_face.xml')



#declare persons and images
person1_image = face_recognition.load_image_file('userpics/ygt.jpg');
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file('userpics/doganay.jpg');
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]


known_face_encodings = [person1_face_encoding,person2_face_encoding]
known_face_names = ["yigit","doganay"]






#initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = camera.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown.."

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)



    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # cv2.putText(frame,"human",(left + 94,bottom - 3),font, 0.5,(255,255,255),1)



    #ISHUMAN CHECK CONTROL
    grey_ton = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Convertcolor(cvtColor)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame2 = cv2.resize(grey_ton, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame2 = small_frame[:, :, ::-1]

    faces = frontal_face_extended.detectMultiScale(rgb_small_frame2,1.1,2)

    for(x,y,w,h) in faces:
        #show frame
        x *= 4
        y *= 4
        w *= 4
        h *= 4
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) #framei göster,sol üst ,sağ üst koordinatla,renk,kalınlık






    #frame = video_capture.read()[1]

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector(rgb_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release handle to the webcam
camera.release()
cv2.destroyAllWindows()



# ----------------

