import cv2
import numpy as np
import os
from datetime import datetime

# Playing video from file:
cap = cv2.VideoCapture(0)

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')
print(cap.get(cv2.CAP_PROP_FPS))
currentFrame = 0
#top
x=0#59
#left
y=0 #71
#right
w=0 #160
#bottom
h=0 #153
flag=1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,1024)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,768)
    cv2.imshow('frame',frame)
    #fromCenter = False
    #r = cv2.selectROI(frame, fromCenter)
    if(flag==1):
        r = cv2.selectROI(frame)
        imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        x=int(r[1])
        y=int(r[0])
        w=int(r[3])
        h=int(r[2])
        print(x,x+w)
        print(y,y+h)
        cv2.imwrite("roi.png",imCrop)
        cv2.destroyWindow("ROI selector")
        flag=0
    
    #print(x,y,x+w,y+h)
    #print(type(r[1]))
    #top=int(r[1])
    #left=int(r[1]+r[3])
    #bottom=int(r[0])
    #right=int(r[0]+r[2])
    #cv2.imwrite("roi.png",r)
    
    #imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.rectangle(frame,(y,x),(y+h,x+w),(0,0,255),10)
    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),6)
    cv2.imshow('frame',frame)
    #cv2.rectangle(frame,(59,71),(160,153),(0,0,255),5)
    #cv2.rectangle(frame, (250,30), (450,200), (0,255,0), 5)
 

    # Display cropped image
    #cv2.imshow("Image", imCrop)
    #cv2.imwrite("roi.png",imCrop)
    #cv2.waitKey(0)
    #if(r==[0,0,0,0]):
        
    #current_time= datetime.now().second
    #print(current_time)
    currentFrame += 1
    #print("just before if")
    #cv2.rectangle("frame",)
    if(currentFrame%30==0):
        #print("inside if")
        name = './data/frame' + str(currentFrame) + '.jpg'
       # print ('Creating...' + name)
        cv2.imwrite(name,frame)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
        
 
    
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""
#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
# Allows use of print like a function in Python 2.x

# Import OpenCV and Numpy modules
import numpy as np
import cv2
 
try:
    # Create a named window to display video output
    cv2.namedWindow('Watermark', cv2.WINDOW_NORMAL)
    # This section is the same from previous Image example.
    # Load logo image
    dog = cv2.imread('E:\\D DRIVDE\\STUDY\\Python Study\\CODES\\door.png')
    # 
    rows,cols,channel = dog.shape
    # Convert the logo to grayscale
    dog_gray = cv2.cvtColor(dog,cv2.COLOR_BGR2GRAY)
    # Create a mask of the logo and its inverse mask
    ret, mask = cv2.threshold(dog_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now just extract the logo
    dog_fg = cv2.bitwise_and(dog,dog,mask = mask)
    
    # Initialize Default Video Web Camera for capture.
    webcam = cv2.VideoCapture(0)
    # Check if Camera initialized correctly
    success = webcam.isOpened()
    if success == False:
        print('Error: Camera could not be opened')
    else:
        print('Sucess: Grabbing the camera')
        webcam.set(cv2.CAP_PROP_FPS,30)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH,1024)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,768)

    while(True):
        # Read each frame in video stream
        ret, frame = webcam.read()
        # Perform operations on the video frames here
        # To put logo on top-left corner, create a Region of Interest (ROI)
        roi = frame[0:rows, 0:cols ] 
        # Now blackout the area of logo in ROI
        frm_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        # Next add the logo to each video frame
        dst = cv2.add(frm_bg,dog_fg)
        frame[0:rows, 0:cols ] = dst
        # Overlay Text on the video frame with Exit instructions
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Type q to Quit:",(50,700), font, 1,(255,255,255),2,cv2.LINE_AA)
        # Display the resulting frame
        cv2.imshow('Watermark',frame)
        # Wait for exit key "q" to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Quitting ...')
            break

    # Release all resources used
    webcam.release()
    cv2.destroyAllWindows()

except cv2.error as e:
    print('Please correct OpenCV Error')
"""
