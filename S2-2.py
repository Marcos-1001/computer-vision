import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
from io import BytesIO



# %% 

def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def binarize_image(image, threshold=128):
    
    binary_image = image.convert('L')  # Convert to grayscale
    
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            pixel = binary_image.getpixel((i, j))
            if pixel > threshold:
                binary_image.putpixel((i, j), 255)
            else:
                binary_image.putpixel((i, j), 0)
    
    
    return binary_image

def otsu_binarization(image):
    
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)    
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return Image.fromarray(binary_image)
    
def masking(image, mask):
    
    image = np.array(image)
    mask = np.array(mask)
    
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return Image.fromarray(masked_image)

def fusion(image1, image2):
    
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    
    # put image1 on top of image2
    fused_image = np.where(image1 != 0, image1, image2)
    # Convert to uint8
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    
    
    return Image.fromarray(fused_image)
# https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_square.jpg
# https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSONYzUJxzKuAwEmaZa_Mfm4-HHrhCyWqpf6g&s
img_1 = get_image('https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_square.jpg')
img_2 = get_image('https://static.vecteezy.com/system/resources/thumbnails/002/098/203/small_2x/silver-tabby-cat-sitting-on-green-background-free-photo.jpg')
img_1 = img_1.resize((300, 300))
img_2 = img_2.resize((300, 300))
img_1_mask = masking(img_1, binarize_image(img_1, 128))

fusion_img = fusion(img_1_mask, img_2)



plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_1)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('New Image')
plt.imshow(fusion_img)
#binarized_img = otsu_binarization(img_1)
#plt.imshow(binarized_img, cmap='gray')
plt.axis('off')
plt.show()
