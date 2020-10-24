import cv2
import sys
import os
import numpy as np
from matplotlib import pyplot as plt

try:
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('number_plates'):
        os.makedirs('number_plates')
except OSError:
    print ('Error: Creating directory of data')
currentFrame = 0
shape = "unidentified"
#top
x=0#59
#left
y=0 #71
#right
w=0 #160
#bottom
h=0 #153
flag=1
imCrop=''
cap=cv2.VideoCapture(filename = "E:\\Parking_Hero\\1.MOV")
while(cap.isOpened()):
    ret, frame= cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',frame)
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
        gray_roi= cv2.cvtColor(imCrop,cv2.COLOR_BGR2GRAY)
        blur_roi = cv2.GaussianBlur(src = gray_roi, ksize = (5, 5), sigmaX = 0)
        t_roi, maskLayer_roi= cv2.threshold(src = blur_roi, thresh = 0,maxval = 255,type = cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
        print(t_roi)
        cv2.destroyWindow("ROI selector")
        
        flag=0
    
   
    
    
    cv2.rectangle(frame,(y,x),(y+h,x+w),(0,0,255),10)
    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),6)
    #cv2.imshow('frame',frame)
   
    currentFrame += 1
   
    if(currentFrame%30==0):
        name = './data/frame' + str(currentFrame) + '.jpg'
        cv2.imwrite(name,frame)
        
    
    cv2.namedWindow(winname = "Grayscale Image", flags = cv2.WINDOW_NORMAL)
    cv2.imshow(winname = "Grayscale Image", mat = gray)
    blur = cv2.GaussianBlur(src = gray, ksize = (5, 5), sigmaX = 0)
    (t, maskLayer) = cv2.threshold(src = blur, thresh = 0,maxval = 255,type = cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
    #print(t)
    #cv2.imshow("Threshold",maskLayer)
 

    if(t>t_roi):
        cv2.rectangle(frame,(y,x),(y+h,x+w),(0,255,0),10)
        if(currentFrame%30==0):
            name2 = './number_plates/p' + str(currentFrame) + '.jpg'
            crop= frame[x:x+w,y:y+h]
            cv2.imwrite(name2,crop)
            
            #crop_gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            im=cv2.imread('E:\\Parking_Hero\\numberplate.jpg')
            crop_gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            blur_crop = cv2.GaussianBlur(src = crop_gray, ksize = (5, 5), sigmaX = 0)
            #(t_crop, maskLayer_crop) = cv2.threshold(src = blur_crop, thresh = 0,maxval = 255,type = cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
            (t_crop, maskLayer_crop) = cv2.threshold(src = blur_crop, thresh = 200,maxval = 255,type = cv2.THRESH_BINARY)
            contours_crop,hierarchy= cv2.findContours(image=maskLayer_crop,mode= cv2.RETR_EXTERNAL,method= cv2.CHAIN_APPROX_SIMPLE)
            
            maskLayer_crop_erode = cv2.erode(maskLayer_crop, None, iterations=1)
            maskLayer_crop = cv2.dilate(maskLayer_crop, None, iterations=4)
            m,n,o,p = cv2.boundingRect(contours_crop[0])
            cv2.rectangle(crop_gray,(m,n),(m+o,n+p),(255,0,0),10)
            cv2.imshow("finallll",maskLayer_crop)
            cv2.imshow("finallll_erode",maskLayer_crop_erode)   
        

        
        
    else:
        cv2.rectangle(frame,(y,x),(y+h,x+w),(0,0,255),10)
    
    
        #cnt= contours[0]
        #ctr = np.array(cnt).reshape(-1,1,2).astype(np.int8)
            #print(type(cnt))
    #print(type(ctr))
        
        #for (i, c) in enumerate(contours):
            #print("\tSize of contour %d: %d" % (i, len(c)))
    #for c in contours:
        # Returns the location and width,height for every contour
     #       x, y, w, h = cv2.boundingRect(c)
      #      print(x,y,w,h)
        
    cv2.imshow("final",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()




"""
import numpy as np
import cv2
import  imutils

# Read the image file
image = cv2.imread('E:\\PROJECTS\\Python_Codes\\number_plates\\p392.jpg')

# Resize the image - change width to 500
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("Original Image", image)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imwrite("gray",gray)
cv2.imshow("1 - Grayscale Conversion", gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("4 - Canny Edges", edged)

# Find contours based on Edges
cnts,hierarchy= cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
#sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None
#we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  # Select the contour with 4 corners
        NumberPlateCnt = approx #This is our approx Number Plate Contour
        break


# Drawing the selected contour on the original image
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)

cv2.waitKey(0) #Wait for user input before closing the images displayed
"""




#--------------EDGE DETECTION-------------# 
"""
import os,glob
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image,ImageFilter
from PIL.ImageFilter import (UnsharpMask)

im=cv2.imread("C:\\Users\\User\\Desktop\\15_1.jpg")
x1=824 
x2=478 
y1=1050 
y2=535 
#r=90-angle
#print("angle",angle)
angle= math.degrees(math.atan2(y2-y1,x2-x1))
if(angle>90):
    angle=angle-90
img_rotated = ndimage.rotate(im,angle)
m=cv2.rectangle(im,(x1,y2),(y1,x2),(0,0,255),10)

#imCrop=im[y:y+h, x:x+w]
imCrop = im[int(x2):int(x2+(y2-x2)), int(x1):int(x1+(y1-x1))]

dst = cv2.fastNlMeansDenoisingColored(imCrop,None,10,10,7,21)

plt.subplot(121),plt.imshow(imCrop)
plt.subplot(122),plt.imshow(dst)
gray_roi= cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
blur_roi = cv2.GaussianBlur(src = gray_roi, ksize = (5, 5), sigmaX = 0)
ret,thresh = cv2.threshold(src = blur_roi, thresh = 0,maxval = 255,type = cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
cv2.imshow("threshafter",thresh)

plt.show()

image = Image.fromarray(imCrop.astype('uint8'))
new_image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
new_image.save("C:\\Users\\User\\Desktop\\unsharped.jpg")

plt.subplot(121),plt.imshow(image, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(new_image, cmap = 'gray')
plt.title('Unsharp Filter'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.imwrite("C:\\Users\\User\\Desktop\\images_15_1.jpg",imCrop)
im2 = image.filter(ImageFilter.CONTOUR)
im2.save("C:\\Users\\User\\Desktop\\filter_contour.jpg")

im2.show()
im3 = image.filter(ImageFilter.EDGE_ENHANCE)  
im3.show()
im3.save("C:\\Users\\User\\Desktop\\edge_enhance.jpg")

im4=image.filter(ImageFilter.EDGE_ENHANCE_MORE)
im4.show()
im4.save("C:\\Users\\User\\Desktop\\edge_enhance_more.jpg")

im5=im4.filter(ImageFilter.SHARPEN)
im5.show()
im5.save("C:\\Users\\User\\Desktop\\sharpen.jpg")

im6=image.filter(ImageFilter.FIND_EDGES)
im6.show()
im6.save("C:\\Users\\User\\Desktop\\find_edges.jpg")

im7 = im4.filter(UnsharpMask(radius=2, percent=150, threshold=3))
im7.show()
im7.save("C:\\Users\\User\\Desktop\\unsharpmask.jpg")

cv2.imwrite('C:\\Users\\User\\Desktop\\rotated.jpg', m)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
retv, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers=markers.astype(np.int32)
#print(type(markers))
#markers = cv2.watershed(im,markers)
#im[markers == -1] = [255,0,0]
#cv2.imshow("markers",markers)

border = cv2.dilate(imCrop, None, iterations=5)
erode=cv2.erode(border, None)
border = border - erode

cv2.imshow("border",border)
cv2.imshow("erode",erode)
"""

"""


import sys
import cv2
import numpy as np
from scipy.ndimage import label

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now. 
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    return 255 - lbl


img = cv2.imread("C:\\Users\\User\\Desktop\\images_15_1.jpg")

# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
_, img_bin = cv2.threshold(img_gray, 0, 255,cv2.THRESH_OTSU)
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,np.ones((3, 3), dtype=int))

result = segment_on_dt(img, img_bin)
cv2.imwrite("C:\\Users\\User\\Desktop\\result.jpg", result)

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
cv2.imwrite("C:\\Users\\User\\Desktop\\img.jpg", img)
"""
