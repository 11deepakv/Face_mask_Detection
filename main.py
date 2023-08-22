import tensorflow as tf
import os
from keras.preprocessing import image
import cv2
from keras.applications.vgg16 import VGG16
from keras import Sequential
from keras.layers import Dense

model = tf.keras.models.load_model('my_model.keras')

def detect_face(img):
    coods = haar.detectMultiScale(img)
    return coods

def draw_label(img, text, pos, bg_color):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, cv2.FILLED)
    end_x = pos[0]+text_size[0][0]+2
    end_y = pos[1]+text_size[0][1]-2
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, cv2.FILLED)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), cv2.LINE_AA)

def detect_face_mask(img):
    y_pred = (model.predict(img.reshape(1,224,224,3))>0.5).astype('int32')
    return y_pred[0][0]

haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # call the detecction method
    img = cv2.resize(frame,(224,224))
    y_pred = detect_face_mask(img)
    
    coods = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    
    for x,y,w,h in coods:
        cv2.rectangle(frame, (x,y),(x+w, y+h), (0,0,255), 1)
    
    if y_pred == 0:
        draw_label(frame, 'Mask', (50,50), (0,0,255))
    
    if y_pred == 1:
        draw_label(frame, 'No Mask', (50,50), (0,0,255))
        
    cv2.imshow('window', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()