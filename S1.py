# %% 
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



    
    
img = get_image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSONYzUJxzKuAwEmaZa_Mfm4-HHrhCyWqpf6g&s')

level = 8
bin_size = 256 // level


def quantize(img, level):
    img_ = np.array(img)
    img_ = img_ // bin_size * bin_size + bin_size // 2
    return Image.fromarray(img_)
    
    
        
img_ = quantize(img, level)


plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img_, cmap='gray')
plt.show()
