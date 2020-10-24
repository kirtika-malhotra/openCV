import numpy as np
import cv2
import os
from PIL import Image
import pickle

#print(cv2.__file__)

#GETTING STARTED WITH IMAGES
# Load an color image in grayscale
img = cv2.imread('image.jpg',0)
#cv2.imshow('image',img)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('image.png',img)
img=cv2.imread("image.jpg",0)
cv2.imshow("image",img)
k=cv2.waitKey(0)
if(k==27):
    cv2.destroyAllWindows()

elif(k==ord('s')):
    cv2.imwrite("image2.png",img)
    cv2.destroyAllWindows()


#GETTING STARTED WITH VIDEOS
"""video=cv2.VideoCapture(0)
while('TRUE'):
    print("inside while")
    #capture frame_by_frame
    ret,frame=video.read()
    cv2.imshow("frame",frame)
    if (cv2.waitKey(0) | ord('q')):
        break
video.release()
cv2.destroyAllWindows()
"""
"""cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
   # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.VideoCapture("frame")
        cv2.waitKey(25)
        break
fourcc= cv2.VideoWriter_fourcc(*'XVID')
out= cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

"""


"""cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
frame_width= int(cap.get(3))
frame_height= int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (frame_width,frame_height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        #frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
"""

#Arithmetic operations on images
"""
img1= cv2.imread('download.jpg')
img2= cv2.imread('download2.jpg')
dst= cv2.addWeighted(img1,0.5,img2,0.3,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
"""
img1= cv2.imread('im1.jpg')
img2= cv2.imread('im2.jpg')
rows,cols,ch= img2.shape
roi= img1[0:rows, 0:cols]
img2gray= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
#Basic Operations on Images
"""
img1= cv2.imread('download.jpg')
img2= cv2.imread('download2.jpg')
px= img1[100,100]
'blue green red values'
print(px)
blue= img1[100,100,0]
print(blue)
print(img1.shape)
print(img1.size)
print(img1.dtype)
part = img1[180:140, 130:190]
img1[273:333, 100:160] = part
'SPLITTING AND MERGING IMAGE channel'
b,g,r= cv2.split(img2)
img2= cv2.merge((b,g,r))
img3=cv2.imread('im12.jpg')
rep= cv2.copyMakeBorder(img3,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
"""

#Performance meaurement and improvement techniques

#IMAGE PROCESSING
"""
cap=cv2.VideoCapture(0)
while(1):
    _,frame= cap.read()
    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue= np.array([110,50,50])
    upper_blue= np.array([130,255,255])
    mask= cv2.inRange(hsv,lower_blue,upper_blue)
    res= cv2.bitwise_and(frame,frame,mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k= cv2.waitKey(5) & 0xFF
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
"""

#geometric transformation of images
"""
img1= cv2.imread('download.jpg')
#res = cv2.resize(img1,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
resize=cv2.resize(img1,(int(img1.shape[1]*2),int(img1.shape[0]*2)))
cv2.imshow('resize',resize)
"""


#Face Detection using opencv
"""face_cascade= cv2.CascadeClassifier("C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python36-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
img= cv2.imread("abc.jpg")
gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces= face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors= 5)
print(type(faces))
print(faces)
for x,y,w,h in faces:
    img= cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
resized = cv2.resize(img, (int(img.shape[1]),int(img.shape[0])))
cv2.imshow("Gray", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#face detection in video
"""face_cascade= cv2.CascadeClassifier("C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python36-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)
while (True):
    ret,frame= cap.read()
    
    gray_img= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)
    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray= gray_img[y:y+h, x:x+w] #[ycord_start:ycord_end, xcord_start:xcord_end]
        print(roi_gray)
        cv2.imwrite("my-img.jpg",roi_gray)
#resized = cv2.resize(img, (int(img.shape[1]),int(img.shape[0])))
    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

#Face detection using my own HAARCASCADE
"""face_cascade= cv2.CascadeClassifier("E:\\D DRIVDE\\STUDY\\haarcascade\\dasar_haartrain\\myhaar.xml")
img= cv2.imread("abc.jpg")
gray_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces= face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors= 5)
print(type(faces))
print(faces)
for x,y,w,h in faces:
    img= cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),stroke=3) #stroke means thickness 
resized = cv2.resize(img, (int(img.shape[1]),int(img.shape[0])))
cv2.imshow("Gray", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#FACE RECOGNITION
""" HOW TO RECOGNIZE? USING DEEP LEARNED MODELS LIKE KERAS,TENSORFLOW,PYTORCH , SCIKIT
NOT SO PERFECT MODEL(KERAS) WE ARE ABOUT TO USE
"""
#open image directory

"""BASE_DIR= os.path.dirname(os.path.abspath(__file__))   #base directory
img_dir= os.path.join(BASE_DIR, "IMAGES")   #to get the IMAGES folder
#E:\\D DRIVDE\\STUDY\\haarcascade\\dasar_haartrain\\myhaar.xml

current_id=0
label_ids={}
face_cascade= cv2.CascadeClassifier("C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python36-32\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml")
#LBPH Face recognizer is created
recognizer= cv2.face.LBPHFaceRecognizer_create()
y_labels=[]
x_train= np.empty
#to see the images in the folder
for roots,dirs,files in os.walk(img_dir):   #to get all images in the image directory
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):    #to get all png,jpg files
            path= os.path.join(roots,file)  #path of the image
            
            #to get the label name and replace the space with - and converts uppper case to lower if present
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            
            if not label in label_ids:
                label_ids[label]=current_id
                current_id += 1
            id_= label_ids[label]
            
            y_labels.append(label)  #we want some  umbers in labels
            x_train= np.append(x_train,path) #verify this image, convert it to numpy array, gray
            pil_image=  Image.open(path,mode='r').convert("L")    #will give me the image of this path and convert image into grayscale
            
            image_array=np.array(pil_image) #turn image to numpy array
            
            
            faces= face_cascade.detectMultiScale(image_array,scaleFactor=1.50,minNeighbors=5)
            
            for (x,y,w,h) in faces:
                roi= image_array[y:y+h, x:x+w]
                
                x_train=np.append(x_train,roi) #training data
                
                y_labels.append(id_)
                #cv2.rectangle(image_array,(x,y),(x+w,y+h),(255,0,0),stroke=5)
                
#print(y_labels)
#c=Image.open(x_train,mode='r')
                #USING PICKLES TO SAVE LABEL IDS
with open("label.pickle",'wb') as f: #create a label.pickle with write binary(wb)
    pickle.dump(label_ids,f)
labels=np.array(y_labels)

#Train Opencv Recognizer
recognizer.update(x_train,labels)  #training and saving y_labels as nparray
recognizer.save("trainner.yml")      #saving the recognizer 

"""



#CONTROL NO. OF FRAMES IN LIVE VIDEO
"""cap = cv2.VideoCapture(0)
fps=0 #for decreasing fps

while True:
    ret,frame=cap.read()
    
    if fps>=10:
        cv2.imshow('frame drop',frame)
        fps=0
    fps+=1
    if(cv2.waitKey(1) | 0xFF==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()



"""


"""
cap=cv2.VideoCapture(0)
while(True):
    ret,frame= cap.read()
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) or 0xFF==ord('q'):
        break
cap.release()
"""

