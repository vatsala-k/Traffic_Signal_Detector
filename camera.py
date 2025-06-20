
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

classifier=load_model(r'C:\Users\SSD\Desktop\project\model.h5')
signal_detector= cv2.CascadeClassifier(r'C:\Users\SSD\Desktop\project\haarcascade_frontalface_default.xml')


framewidth=640
frameheight=480
brightness=180
threshold=0.90
font=cv2.FONT_HERSHEY_SIMPLEX

cap=cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4,frameheight)
cap.set(10,brightness)


def grayscale(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img=cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img=grayscale(img)
    img= equalize(img)
    img=img/255
    return img
def getclassname(classNo):
    if(classNo==0):
        return "Speed limit (20km/h)"
    elif(classNo==1):
        return "Speed limit (30km/h)"
    elif(classNo==2):
        return "Speed limit (50km/h) "
    elif(classNo==3):
        return "Speed limit (60km/h)"
    elif(classNo==4):
        return "Speed limit (70km/h)"
    elif(classNo==5):
        return "Speed limit (80km/h) "
    elif(classNo==6):
        return "End of speed limit (80km/h) "
    elif(classNo==7):
        return "Speed limit (100km/h)"
    elif(classNo==8):
        return "Speed limit (120km/h)"
    elif(classNo==9):
        return "No passing"
    elif(classNo==10):
        return "No passing for vechiles over 3.5 metric tons"
    elif(classNo==11):
        return "Right-of-way at the next intersection"
    elif(classNo==12):
        return "Priority road"
    elif(classNo==13):
        return "Yield"
    elif(classNo==14):
        return "Stop"
    elif(classNo==15):
        return "No Vehicles"
    elif(classNo==16):
        return "Vehicles over 3.5 metric tons prohibited"
    elif(classNo==17):
        return "No Entry"
    elif(classNo==18):
        return "General Caution"
    elif(classNo==19):
        return "Dangerous curve to the left"
    elif(classNo==20):
        return "Dangerous curve to the right"
    elif(classNo==21):
        return "Double curve"
    elif(classNo==22):
        return "Bumpy road"
    elif(classNo==23):
        return "Slippery road"
    elif(classNo==24):
        return "Road narrows on the right"
    elif(classNo==25):
        return "Road work"
    elif(classNo==26):
        return "Traffic signals"
    elif(classNo==27):
        return "Pedestrian"
    elif(classNo==28):
        return "Children crossing"
    elif(classNo==29):
        return "Bicycle crossing "
    elif(classNo==30):
        return "Beware of ice"
    elif(classNo==31):
        return "Wild animals crossing"
    elif(classNo==32):
        return "End of all speed and passing limits"
    elif(classNo==33):
        return "Turn right ahead"
    elif(classNo==34):
        return "Turn left ahead"
    elif(classNo==35):
        return "Ahead only"
    elif(classNo==36):
        return "Go straight or right"
    elif(classNo==37):
        return "Go straight or left"
    elif(classNo==38):
        return "Keep right"
    elif(classNo==39):
        return "Keep left"
    elif(classNo==40):
        return "Roundabout mandatory"
    elif(classNo==41):
        return "End of no passing"
    elif(classNo==42):
        return "End of no passing by vechiles over 3.5 metric tons"
    

while(True):
    success, imgOriginal= cap.read()

    img=np.asarray(imgOriginal)
    img=cv2.resize(img,(32,32))
    img=preprocessing(img)
    cv2.imshow("Processed image", img)
    img=img.reshape(1,32,32,1)
    cv2.putText(imgOriginal, "class", (20, 35), font, 0.75, (255, 255, 255), 2)
    predictions=classifier.predict(img)
    classIndex = np.argmax(classifier.predict(img), axis=-1)
    probabilityvalue=np.amax(predictions)
    if probabilityvalue > threshold:
        cv2.putText(imgOriginal,str(classIndex)+" "+str(getclassname(classIndex)),(120,35),font,0.75,(0,0,255),2, cv2.LINE_AA)
        cv2.putText(imgOriginal,str(round(probabilityvalue*100,2))+"%",(180,75),font,0.75,(0,0,255),2, cv2.LINE_AA)
    cv2.imshow("result",imgOriginal)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
