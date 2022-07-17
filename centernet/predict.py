import time
import cv2
import numpy as np
from PIL import Image

from centernet import CenterNet

if __name__ == "__main__":
    centernet = CenterNet()
    mode = "predict"
    crop            = False

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = centernet.detect_image(image, crop = crop)
                r_image.show()
                r_image.save("1.jpg", quality=95, subsampling=0)
