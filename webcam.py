import cv2
import mediapipe as mp
import os
import time
import posemodule as pm
import math
import random


colors = [(245,117,16), (117,245,16), (16,117,245)]
pTime = 0
windowname = "OpenCV Media Player"
cv2.namedWindow(windowname)
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count = 0
count70 = 0

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def check(a, b):
    if a > b:
        return a-b
    else:
        return b-a



def prob_viz(num, input_frame, colors):
    output_frame = input_frame.copy()
    message = "";
    colr = 0
    if num>=90:
        colr = 1
        message = "Progress {}%".format(str(num))
    elif num<85 and num>=50:
        colr  = 0
        message = "Progress {}%".format(str(num))
    elif num < 50:
        colr = 2
        message = "Start"
    
    cv2.rectangle(output_frame, (50,15), (num+110, 23), colors[colr], 13)
    cv2.putText(output_frame, message, (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
     
    return output_frame



f=0
k=0
time.sleep(5)
cap = cv2.VideoCapture(0)
img = cap.read()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmlist = detector.getPosition(img,draw=False)
    
    if len(lmlist)!=0:
        cv2.circle(img,(lmlist[25][1],lmlist[25][2]),8,(255,150,0),cv2.FILLED)
        cv2.circle(img,(lmlist[23][1],lmlist[23][2]),8,(255,150,0),cv2.FILLED)
         #print(lmlist[23])
        y1 = lmlist[25][2]
        y2 = lmlist[23][2]
        
        #length = a-b
        length = y2-y1
        if length>=-45 and f==0:
            f=1
        elif length<-50 and f==1:
            f=0
            count=count+1
            count70=count70-1
        elif length>=-57 and k==0:
            k=1
        elif length<-60 and k==1:
            k=0
            count70=count70+1
        
        print("Value of Y1  = {}".format(y1))
        print("Value of Y2  = {}".format(y2))
        print("Value of Length  = {}".format(length))
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,"100% " + "Total Number of Squats  "+str(int(count)),(50,60),cv2.FONT_HERSHEY_DUPLEX,0.5,
        (60,100,255),1)
        cv2.putText(img,"Calories Burnt  "+str(int(count)*0.32),(50,140),cv2.FONT_HERSHEY_DUPLEX,0.5,
        (60,100,255),1)
        #img = cv2.resize(img, (900,900))                    # Resize image
        
        
        xx = abs(length)
        progress = 0
        different = xx - 50
        print("Different value = {}".format(different))
        if different > 30:
            progress = 10
        elif different <= 25 and different > 20:
            progress = 30
        elif different <= 20 and different > 15:
            progress = 50
        elif different <= 15 and different > 10:
            progress = 60
        elif different <= 10 and different > 5:
            progress = 70
        elif different <= 5 and different > 2:
            progress = 90
        elif different <= 2 and different <=0:
            progress = 100
            
        img = prob_viz(progress , img, colors)
        
        print("xx value = {}".format(xx))
        print("-------------------------------progress value = {}".format(progress))
        
        #count70 = check(count, count70)
        cv2.putText(img,"70% " + "Total Number of Squats  "+str(int(count70)),(50,100),cv2.FONT_HERSHEY_DUPLEX,0.5,
        (60,100,255),1)
        
        cv2.imshow(windowname,img)
        calories = 0.32*count
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
cap.release()
cv2.destroyAllWindows()

