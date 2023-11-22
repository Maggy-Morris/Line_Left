import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import os 
from datetime import datetime 
import csv
import pyrebase

model=YOLO('yolov8s.pt')


firebaseConfig = {
  "apiKey": "AIzaSyAGU1uOD4K3Aje2UqYbW6HrHQGe98aVXV0",
  "authDomain": "shsh-3fec7.firebaseapp.com",
  "databaseURL": "https://shsh-3fec7-default-rtdb.firebaseio.com",
  "projectId": "shsh-3fec7",
  "storageBucket": "shsh-3fec7.appspot.com",
  "messagingSenderId": "52920915309",
  "appId": "1:52920915309:web:a6205594f46e0eeac2b664"
}



firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
db = firebase.database()


# Get the directory of the currently executing script
script_directory = os.path.dirname(__file__)

# Set path in which you want to save images and CSV file
path = script_directory

# Set path in which you want to save images and CSV file

csv_file = os.path.join(path, 'database.csv')


# Changing directory to the given path 
os.chdir(path) 

# Variable to give a unique name to images 
i = 1

wait = 0


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture(0 , cv2.CAP_DSHOW)

current_time = str(datetime.now())



my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count=0
persondown={}
tracker=Tracker()
counter1=[]

personup={}
counter2=[]
cy1=230
cy2=220
offset=6







while True:    
    ret,frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list=[]
   
    for index,row in px.iterrows():
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:

            list.append([x1,y1,x2,y2])
       
        
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.circle(frame,(cx,cy-10),4,(255,0,255),-1)
        if cy1 <(cy + offset ) and cy1 >(cy-offset):
          cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255))
          cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
          persondown[id] =(cx,cy)
          if id in persondown :
            if cy2 <(cy + offset ) and cy2 >(cy-offset):
                 cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255))
                 cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                 if counter1.count(id) == 0 :
                  counter1.append(id)
                  #####for up 
          if cy1 <(cy + offset ) and cy1 >(cy-offset):
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255))
            cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
            personup[id] =(cx,cy)
          if id in personup :
            if cy1 <(cy + offset ) and cy1 >(cy-offset):
                 cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,255))
                 cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
            if counter2.count(id) == 0 :
                  counter2.append(id)         

        
    cv2.line(frame,(3,cy1),(1018,cy1),(0,255,0),2)
    
    wait = wait + 1000
    current_time = str(datetime.now())

    up=(len(counter2))
    
     # When it reaches 5000 milliseconds, save the frame and timestamp in Firebase
    if wait == 100000: 
        filename = 'Left_Gate '+datetime.now().strftime('%H-%M-%S')+'.jpg'
            
        # Save the images in the given path 
        cv2.imwrite(os.path.join(path, filename), frame) 

        # Upload frame image to Firebase Storage
        storage.child(filename).put(filename)
        current_time = str(datetime.now())
        
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame', 'Timestamp','Total_people_In'])  # Write headers to the CSV file
        
            writer.writerow(['Left_Gate '+datetime.now().strftime('%H-%M-%S'), current_time,up])


        # Write frame number and timestamp to the Realtime Database
        db.child("timestamps").child(f"Frame_{datetime.now().strftime('%H-%M-%S')}").push({
            "Frame": 'Left_Gate '+datetime.now().strftime('%H-%M-%S'),
            "Timestamp": current_time,
            "Total_people_In" : up
        })    
        i += 1
        wait = 0

        
 
    cvzone.putTextRect(frame,f'IN = {up}',(50,60),2,2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

