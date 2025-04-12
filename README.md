# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the required libraries and images for the experiment

### Step2:
Translate the image using wrapAffine and the same can be used for Shearing the image

### Step3:
Use cv2.resize() to scale the image

### Step4:
use cv2.flip(img,2) to get the reflected image

### Step5:
Crop the image by using slicing

## Program:
```python
Developed By: Priyadharshan
Register Number: 212223240127

i)Image Translation

import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('cologne.jpg')
colonge_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(colonge_img)
plt.title("cologne Cathedral")  
plt.axis('off')

tx, ty = 100, 50
M_translation = np.float32([[2, 0, tx], [0, 2, ty]])  
translated_img = cv2.warpAffine(img, M_translation, (img.shape[1], img.shape[0]))

plt.imshow(cv2.cvtColor(translated_img, cv2.COLOR_BGR2RGB))  
plt.axis('off')

ii) Image Scaling

fx, fy = 5.0, 2.0  
scaled_img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

plt.imshow(cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB))  
plt.title("Scaled img")  
plt.axis('off')

iii)Image shearing
shear_matrix = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
sheared_img = cv2.warpAffine(img, shear_matrix, (img.shape[1], img.shape[0]))

plt.imshow(cv2.cvtColor(sheared_img, cv2.COLOR_BGR2RGB))  
plt.title("Sheared img") 
plt.axis('off')


iv)Image Reflection

reflected_img = cv2.flip(img, 2)

plt.imshow(cv2.cvtColor(reflected_img, cv2.COLOR_BGR2RGB)) 
plt.title("Reflected img")  
plt.axis('off')

v)Image Rotation

(height, width) = img.shape[:2]  
angle = 45  
center = (width // 2, height // 2)  
M_rotation = cv2.getRotationMatrix2D(center, angle, 1)  

rotated_img = cv2.warpAffine(img, M_rotation, (width, height))

plt.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)) 
plt.title("Rotated img")  
plt.axis('off')


vi)Image Cropping

x, y, w, h = 500, 1600, 700, 850  
cropped_img = img[y:y+h, x:x+w]

plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))  
plt.title("Cropped img")  
plt.axis('off')

```
## Output:
### i)Image Translation
![image](https://github.com/user-attachments/assets/ba7cf5e7-8ce0-465d-b200-a2f293807fc1)

### ii) Image Scaling
![image](https://github.com/user-attachments/assets/7f984700-277c-49bd-a59a-81fea2a8fc03)

### iii)Image shearing
![image](https://github.com/user-attachments/assets/69eb6f46-d992-4d38-899c-d9f462adcbf1)

### iv)Image Reflection
![image](https://github.com/user-attachments/assets/2f933490-e162-4c51-a371-d198e5c87409)

### v)Image Rotation
![image](https://github.com/user-attachments/assets/65971fdb-0ff5-47e6-add2-b09c40e6fb0d)

### vi)Image Cropping
![image](https://github.com/user-attachments/assets/1ece29dc-dd1e-49b2-8e2c-1b545fe48a44)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
