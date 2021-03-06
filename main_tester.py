import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
import face_recognition
from statistics import mode
from datasets.datasets import get_labels
from datasets.inference import draw_bounding_box
from datasets.inference import draw_text
from datasets.preprocessor import preprocess_input
from datasets.inference import apply_offsets


frontal_face_extended=cv2.CascadeClassifier('src/haarcascade_frontalface_default.xml')

USE_WEBCAM = True

emotion_model_path = './src/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')


#----------------

def face_compare(frame,process_this_frame):

    #grey_ton = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #Convertcolor(cvtColor)


    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) 

    #bgr to rgb
    rgb_small_frame = small_frame[:, :, ::-1]


    
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    return face_names
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/10 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        #cv2.rectangle(frame, (left, bottom+36), (right, bottom), (0, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom+20), font, 0.3, (255, 255, 255), 1)




#---------------------


#ISHUMAN CHECK CONTROL
def humanFaceDetect(frame):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)

    #  # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame2 = rgb_small_frame[:, :, ::-1]

    faces = frontal_face_extended.detectMultiScale(rgb_small_frame2,1.1,2)

    for(x,y,w,h) in faces:
        #show frame
        x *= 5
        y *= 5
        w *= 5
        h *= 5
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #framei göster,sol üst ,sağ üst koordinatla,renk,kalınlık



#---------------------



# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (10, 50)

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []



#DECLARE PERSONS

# Load a sample picture and learn how to recognize it.
person1_image = face_recognition.load_image_file("userpics/ygt.jpg")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

#Load a second sample picture and learn how to recognize it.
person2_image = face_recognition.load_image_file("userpics/doganay.jpg")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]


person3_image = face_recognition.load_image_file("userpics/fatmanasenturk.jpg")
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]


person4_image = face_recognition.load_image_file("userpics/bpitt.jpg")
person4_face_encoding = face_recognition.face_encodings(person4_image)[0]

person5_image = face_recognition.load_image_file("userpics/ldicaprio.jpeg")
person5_face_encoding = face_recognition.face_encodings(person5_image)[0]

person6_image = face_recognition.load_image_file("userpics/skotat.jpeg")
person6_face_encoding = face_recognition.face_encodings(person6_image)[0]

person7_image = face_recognition.load_image_file("userpics/sumutcakir.jpg")
person7_face_encoding = face_recognition.face_encodings(person7_image)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding,
    person3_face_encoding,
    person4_face_encoding,
    person5_face_encoding,
    person6_face_encoding,
    person7_face_encoding


]

known_face_names = [
    "Yigit Kader",
    "doganay",
    "Fatmana Senturk",
    "Brad Pitt",
    "Leo Dicaprio",
    "Sezai Tokat",
    "Sevket Cakir"
]



# Initialize some variables for we need
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('testvdo.mp4') 


while cap.isOpened(): # True:
    ret, frame = cap.read()


    humanFaceDetect(frame)


    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    faces = detector(rgb_image)

    face_name = face_compare(rgb_image,process_this_frame)
    for face_coordinates, fname in zip(faces,face_name):

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
            color = (255,0,0)
        elif emotion_text == 'sad':
            color = (112,128,144)
        elif emotion_text == 'happy':
            color = (255,255,0)
        elif emotion_text == 'surprise':
            color = (0,0,255)
        elif emotion_text == 'neutral':
            color = (255,255,0)
        else:
            color = (255,255,0)


        if fname == "Unknown":
            name = str(fname) +" is "+ str(emotion_text)
        else:
            name = str(fname) + " is " + str(emotion_text)



        draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
        draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, name,
                  color, 0, -20, 1.3, 1)




    frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
