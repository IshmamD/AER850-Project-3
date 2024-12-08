# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:39:04 2024

@author: ishma
"""
#STEP 1 - Object Masking

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

og_img1 = cv.imread('./data/motherboard_image.JPEG')
img1 = cv.imread('./data/motherboard_image.JPEG', cv.IMREAD_GRAYSCALE)
_, th_img1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
edges = cv.Canny(th_img1,1,200)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(th_img1, cmap='gray')
plt.title('Adaptive Thresholding'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
plt.show()

contours, hierarchy = cv.findContours(th_img1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


filtered_contours = []

for cnt in contours:
    if cv.contourArea(cnt) > 2000:  # Check if the contour's area is greater than 5000
        filtered_contours.append(cnt)

mask = np.zeros_like(img1)
cv.drawContours(mask, filtered_contours, -1, (255), thickness=cv.FILLED)

plt.figure(figsize=(5, 5))
plt.imshow(mask, cmap='gray')
plt.title('Filtered Contours Mask')
plt.xticks([]), plt.yticks([])
plt.show()

cv.imwrite('./data/pcb_mask.png', mask)

newimg1 = cv.bitwise_and(og_img1, og_img1, mask=mask)
newimg1 = cv.cvtColor(newimg1, cv.COLOR_BGR2RGB)

plt.figure(figsize=(15, 15))
plt.imshow(newimg1)
plt.title('Extracted Motherboard Image')
plt.xticks([]), plt.yticks([])
plt.show()

#STEP 2 - YoloV8 Training - THIS PART WAS DONE IN GOOGLE COLLAB
'''
path = './data/data.yaml'
model = YOLO('yolov8n.pt')

result = model.train(data=path,epochs=60,batch=4,imgsz=900,name='detector')
model.save('detector.pt')
'''

model = YOLO('detector.pt')

ardmega = './data/evaluation/ardmega.jpg'
arduno = './data/evaluation/arduno.jpg'
rasppi = './data/evaluation/rasppi.jpg'

results1 = model.predict(source=ardmega, save=True)

for result in results1:
    im_bgr = result.plot(font_size = 90,pil=True)
    im_rgb = cv.cvtColor(im_bgr, cv.COLOR_BGR2RGB)
    
    plt.figure(figsize=(20, 20))
    plt.imshow(im_rgb)
    plt.title('Predictions for Image 1')
    plt.axis('off')
    plt.show()
    
results2 = model.predict(source=arduno, save=True)

for result in results2:
    im_bgr = result.plot(font_size = 25,pil=True)
    im_rgb = cv.cvtColor(im_bgr, cv.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(im_rgb)
    plt.title('Predictions for Image 2')
    plt.axis('off')
    plt.show()
    
results3 = model.predict(source=rasppi, save=True)

for result in results3:
    im_bgr = result.plot(font_size = 50,pil=True)
    im_rgb = cv.cvtColor(im_bgr, cv.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(im_rgb)
    plt.title('Predictions for Image 3')
    plt.axis('off')
    plt.show()
