
import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
from io import BytesIO




def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img



    
    
img = get_image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSONYzUJxzKuAwEmaZa_Mfm4-HHrhCyWqpf6g&s')
img_np = np.array(img)

k1 = 9
kernel  = np.ones((k1,k1), np.float32)/(k1*k1)

img_conv = cv2.filter2D(img_np, -1, kernel)


k2 = 4
butterworth_filter = np.zeros((k1,k1), np.float32)
for i in range(k2):
    for j in range(k2):
        butterworth_filter[i,j] = 1 / (1 + ((i - k2//2)**2 + (j - k2//2)**2) / (k2//2)**2)
butterworth_filter = butterworth_filter / np.sum(butterworth_filter)

img_conv2 = cv2.filter2D(img_np, -1, butterworth_filter)

# gaussian filter
kernel = cv2.getGaussianKernel(ksize=5, sigma=1)
img_conv3 = cv2.filter2D(img_np, -1, kernel)
# median filter
img_conv4 = cv2.medianBlur(img_np, ksize=3)
# laplacian filter
laplacian_filter = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], dtype=np.float32)

img_conv5 = cv2.filter2D(img_np, -1, laplacian_filter)


# plot 5 images
plt.figure(figsize=(10, 10))
plt.subplot(231)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(232)
plt.imshow(img_conv)
plt.title('Mean Filter')
plt.subplot(233)
plt.imshow(img_conv2)
plt.title('Butterworth Filter')
plt.subplot(234)
plt.imshow(img_conv3)
plt.title('Gaussian Filter')
plt.subplot(235)
plt.imshow(img_conv4)
plt.title('Median Filter')
plt.subplot(236)
plt.imshow(img_conv5)
plt.title('Laplacian Filter')
plt.tight_layout()
plt.show()
