import numpy as np
import cv2
import pickle

#############################################
framewidth = 640
frameheight = 480
brightness = 180
threshold = 0.90
font = cv2.FONT_HERSHEY_SIMPLEX

###############################################
# Set up the video camera
cap = cv2.VideoCapture(0)
cap.set(3, framewidth)
cap.set(4, frameheight)
cap.set(10, brightness)
# Import the trained model
pickle_in = open('model_trained.p', "rb")
model = pickle.load(pickle_in)

def grayscale(img):
    img = cv2.COLOR_BGR2GRAY(img)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def GetClassName (ClassNo):
    if ClassNo == 0 : return 'Speed Limit 20km/h'
    elif ClassNo == 1 : return 'Speed Limit 40km/h'
    elif ClassNo == 2 : return 'Speed Limit 60km/h'

while True:
    # Read image
    success, Originalimg = cap.read()

    # Process image
    img = np.asarray(Originalimg)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(Originalimg, "CLASS :", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(Originalimg, "PROBABILITY :", (20, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    # Predict image
    predictions = model.predict(img)
    classindex = model.predict_classes(img)
    probabilityvalue = np.amax(predictions)
    if probabilityvalue > threshold :
        # print (getClassName(classindex))
        cv2.putText(Originalimg, str(classindex)+" "+str(GetClassName(classindex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(Originalimg, str(round(probabilityvalue*100, 2))+"%", (100, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result", Originalimg)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
























