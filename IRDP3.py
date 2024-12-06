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

#largest_contour = max(contours, key=cv.contourArea)

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

#STEP 2 - YoloV8 Training

path = './data/train'
model = YOLO('yolov8n.pt')

result = model.train(data=path,epochs=50,batch=32,imgsz=900,name='detector')
model.save('detector.pt')