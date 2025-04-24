import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def convert_img(img_name):
    grey_img = Image.open("images/" + img_name).convert('L')
    return grey_img


def scale_img(img):
    # push pixel range from 0 to 1 (for stability)
    img_min, img_max = img.min(), img.max()
    scaled_img = (img - img_min) / (img_max - img_min)
    return img_min,img_max, scaled_img


def unscale_img(img, img_min, img_max):
    return (img * (img_max - img_min)) + img_min


if __name__ == '__main__':
    img_name = "man.png"
    grey_img = convert_img(img_name)

    result = grey_img.filter(ImageFilter.GaussianBlur(radius = 2))
    
    plt.imshow(result, cmap='gray')
    plt.savefig(f"results/gaussian_{img_name}")
    plt.axis('on')
    plt.show()
