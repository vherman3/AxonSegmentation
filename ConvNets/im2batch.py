from scipy.misc import imread
import numpy as np
from skimage import exposure
from skimage.transform import rescale


def im2batch(path_image,size, rescale_coeff = 1.0):

    img = imread(path_image, flatten=False, mode='L')
    img = exposure.equalize_hist(img)
    img = rescale(img, rescale_coeff)
    img = img*255
    img = img.astype(int)
    h, w = img.shape

    q_h, r_h = divmod(h, size)
    q_w, r_w = divmod(w, size)

    r2_h = size-r_h
    r2_w = size-r_w
    q2_h = q_h + 1
    q2_w = q_w + 1

    q3_h, r3_h = divmod(r2_h, q_h)
    q3_w, r3_w = divmod(r2_w, q_w)

    dataset = []
    positions=[]
    pos = 0
    while pos+size<=h:
        pos2 = 0
        while pos2+size<=w:
            patch = img[pos:pos+size, pos2:pos2+size]
            dataset.append(patch)
            positions.append([pos,pos2])
            pos2 = size + pos2 - q3_w
            if pos2 + size > w :
                pos2 = pos2 - r3_w

        pos = size + pos - q3_h
        if pos + size > h:
            pos = pos - r3_h

    return [img, np.asarray(dataset), positions]


def batch2im(predictions, positions, h_size, w_size):
    image = np.zeros((h_size, w_size))
    for pred, pos in zip(predictions, positions):
        image[pos[0]:pos[0]+256,pos[1]:pos[1]+256] = pred
    return image

