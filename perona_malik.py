import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def convert_img_to_greyscale(img_name):
    grey_img = Image.open("images/" + img_name).convert('L')
    return np.array(grey_img)


def scale_img(img):
    # push pixel range from 0 to 1 (for stability)
    img_min, img_max = img.min(), img.max()
    scaled_img = (img - img_min) / (img_max - img_min)
    return img_min,img_max, scaled_img


def unscale_img(img, img_min, img_max):
    return (img * (img_max - img_min)) + img_min


def pm_func1(grad_u, k):
    return np.exp(-(np.power(grad_u, 2)) / (np.power(k, 2)))


def pm_func2(grad_u, k):
    return 1 / (1 + np.power((grad_u / k), 2))


def compute_gradients(img):
    # how to compute gradients:
    # https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
    
    center_pixels = img[1:-1, 1:-1]
    # North: I(x, y - 1) = I[i-1, j] - I[i, j]
    north = img[:-2, 1:-1] - img[1:-1, 1:-1]

    # South: I(x, y + 1) = I[i+1, j] - I[i, j]
    south = img[2:, 1:-1] - img[1:-1, 1:-1]

    # East: I(x + 1, y) = I[i, j + 1] - I[i, j]
    east = img[1:-1, 2:] - img[1:-1, 1:-1]

    # West: I(x - 1, y) = I[i, j - 1] - I[i, j]
    west = img[1:-1, :-2] - img[1:-1, 1:-1]
        
    return center_pixels, north, south, east, west


def perona_malik(img, iterations = 10, k = 0.15, lmbd = 0.25):

    img_frame = np.zeros(img.shape, img.dtype)
    for _ in range(iterations):
        center_pixels, dn, ds, de, dw = compute_gradients(img)
        # discretization: https://acme.byu.edu/00000179-afb2-d74f-a3ff-bfbb15700001/anisotropic-pdf
        img_frame[1:-1, 1:-1] = center_pixels + lmbd * ((pm_func1(dn, k) * dn) + (pm_func1(ds, k) * ds) + (pm_func1(de, k) * de) + (pm_func1(dw, k) * dw))
        img = img_frame
    
    return img



if __name__ == '__main__':
    img_name = "man.png"
    grey_img = convert_img_to_greyscale(img_name)
    img_min, img_max, scaled_img = scale_img(grey_img)
    scaled_result = perona_malik(scaled_img, 30, lmbd=0.1)
    result = unscale_img(scaled_result, img_min, img_max)

    plt.imshow(result, cmap='gray')
    # plt.imshow(scaled_result, cmap='gray')
    plt.savefig(f"results/pm_{img_name}")
    plt.axis('on')
    plt.show()

    # Image.fromarray(result).save("pm.jpg")
