import requests
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageOps
from io import BytesIO

def modify_brightness(img, brightness):
    return np.clip(img + brightness, 0, 255).astype(np.uint8)

def contrast(img, contrast):
    return np.clip((img - 128) * contrast + 128, 0, 255).astype(np.uint8)

def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

img = get_image('https://media.istockphoto.com/id/1361394182/photo/funny-british-shorthair-cat-portrait-looking-shocked-or-surprised.jpg?s=612x612&w=0&k=20&c=6yvVxdufrNvkmc50nCLCd8OFGhoJd6vPTNotl90L-vo=')



img = np.array(img)



img = modify_brightness(img, 0)
#img = contrast(img, .4)

histogram = np.zeros((256,), dtype=int)

for row in img:
    for pixel in row:
        histogram[pixel] += 1
histogram = histogram / histogram.sum()


plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(img)
plt.title('Original Image')
plt.subplot(122)
plt.bar(range(256), histogram, width=1)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim(0, 255)
plt.show()