"""import cv2
import numpy as np
import os
cap=cv2.VideoCapture('E:\\Parking_Hero\\1.MOV')


#cv2.imshow('Roi',roi)
#cv2.waitKey(0)
#pix_range=roi_img.size
pix_range=0
#print(pix_range)
#cv2.roiSelector
try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError:
    print ('Error: Creating directory of data')
print(cap.get(cv2.CAP_PROP_FPS))
currentFrame = 0
r=''
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
    cv2.imshow('frame',frame)
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
        pix_range=imCrop.size
        print(pix_range)
        cv2.destroyWindow("ROI selector")
        flag=0
    
    #rows,cols,channels= roi_img.shape
    #roi = roi_img[0:rows, 0:cols]
    
    cv2.rectangle(frame,(y,x),(y+h,x+w),(0,255,0),10)
    
    #cv2.rectangle(frame,(y,x),(y+h,x+w),(0,0,255),10)
        #if pixel value is above 100 then it is converted to 255
    #retval, threshold= cv2.threshold(frame,100, 255,cv2.THRESH_BINARY)
    #retval, threshold= cv2.threshold(frame, pix_range, 255, cv2.THRESH_BINARY)
    
    
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    roi_img= cv2.imread('E:\\Parking_Hero\\roi_img.png',0)
    retval, threshold= cv2.threshold(gray, 100, 255, 0)
    #print(threshold)
    count=cv2.countNonZero(roi_img)
    #print(count)
    #retval2, threshold2= cv2.threshold(gray,100, 255,cv2.THRESH_BINARY)
    #gaus= cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow('frame',frame)
    cv2.imshow('threshold',threshold)
    contours,hierarchy = cv2.findContours(threshold, 1, 2)
    cnt = contours[0]
    area= cv2.contourArea(cnt)
    print(area)
    contours2,hierarchy2 = cv2.findContours(roi_img, 1, 2)
    cnt2 = contours2[0]
    area2= cv2.contourArea(cnt2)
    print(area2)
    if(area>area2):
        print("in if")
        cv2.rectangle(frame,(y,x),(y+h,x+w),(0,0,255),20)
    #cv2.imshow('threshold2',threshold2)
    #cv2.imshow('gaus',gaus)
    #current_time= datetime.now().second
    #print(current_time)
    currentFrame += 1
    
    if(currentFrame%30==0):
        #print("inside if")
        name = './data/frame' + str(currentFrame) + '.jpg'
       # print ('Creating...' + name)
        cv2.imwrite(name,frame)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
#print(area)
#cv2.rectangle(roi,)
"""

"""
import cv2
import sys
from matplotlib import pyplot as plt

# read image, based on command line filename argument;
# read the image as grayscale from the outset
image = cv2.imread(filename = "E:\\Parking_Hero\\test.jpg", flags = cv2.IMREAD_GRAYSCALE)
k = int(sys.argv[2])
t = int(sys.argv[3])
# display the image
cv2.namedWindow(winname = "Grayscale Image", flags = cv2.WINDOW_NORMAL)
cv2.imshow(winname = "Grayscale Image", mat = image)
cv2.waitKey(delay = 0)
# create the histogram
histogram = cv2.calcHist(images = [image], 
    channels = [0], 
    mask = None, 
    histSize = [256], 
    ranges = [0, 256])
# configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0, 256]) # <- named arguments do not work here

plt.plot(histogram) # <- or here
plt.show()
# blur and grayscale before thresholding
#blur = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(src = image, 
    ksize = (k, k), 
    sigmaX = 0)
# perform inverse binary thresholding 
(t, maskLayer) = cv2.threshold(src = blur, 
    thresh = t, 
    maxval = 255, 
    type = cv2.THRESH_BINARY_INV)
 #make a mask suitable for color images(Blue,Green,Red)
mask = cv2.merge(mv = [maskLayer, maskLayer, maskLayer])

# display the mask image
cv2.namedWindow(winname = "mask", flags = cv2.WINDOW_NORMAL)
cv2.imshow(winname = "mask", mat = mask)
cv2.waitKey(delay = 0)
# use the mask to select the "interesting" part of the image
sel = cv2.bitwise_and(src1 = image, src2 = mask)

# display the result
cv2.namedWindow(winname = "selected", flags = cv2.WINDOW_NORMAL)
cv2.imshow(winname = "selected", mat = sel)
cv2.waitKey(delay = 0)

"""
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

# read image, based on command line filename argument;
# read the image as grayscale from the outset
im=cv2.imread(filename = "E:\\Parking_Hero\\car.png")
image = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#k = int(sys.argv[2])
#t = int(sys.argv[3])
# display the image
cv2.namedWindow(winname = "Grayscale Image", flags = cv2.WINDOW_NORMAL)
cv2.imshow(winname = "Grayscale Image", mat = image)

# create the histogram
histogram = cv2.calcHist(images = [image], 
    channels = [0], 
    mask = None, 
    histSize = [256], 
    ranges = [0, 256])
# configure and draw the histogram figure
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0, 256]) # <- named arguments do not work here

plt.plot(histogram) # <- or here
plt.show()

blur = cv2.GaussianBlur(src = image, 
    ksize = (5, 5), 
    sigmaX = 0)
(t, maskLayer) = cv2.threshold(src = blur, 
    thresh = 0,maxval = 255,type = cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
print(t)
cv2.imshow("threshold",maskLayer)
if(t>124):
    print("inside if")
    #cv2.rectangle(image,(y,x),(y+h,x+w),(0,255,0),10)
    contours= cv2.findContours(image=maskLayer,mode= cv2.RETR_EXTERNAL,method= cv2.CHAIN_APPROX_SIMPLE)
    print(type(contours))
    
    cnt= contours[0]
    #ctr = np.array(cnt).reshape(-1,1,2).astype(np.int8)
    print(type(cnt))
    #print(type(ctr))
    for (i, c) in enumerate(contours):
        print("\tSize of contour %d: %d" % (i, len(c)))
    #for cc in contours:
        #nn=0.01*cv2.arcLength(cc,True)
        #approx = cv2.approxPolyDP(cc,nn,True)
        #if(len(approx==4)):
        #    cv2.drawContours(im,cc,0,(0,0,255),-1)
cv2.drawContours(im, contours = cnt, contourIdx = -1, color = (0, 0, 255), thickness = 5)
cv2.imshow("im",im)




