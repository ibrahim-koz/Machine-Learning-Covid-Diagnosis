import random
import cv2
# Returns index of x in arr if present, else -1
import numpy as np


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def shuffle_together(a, b):
    import numpy as np

    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)

    a = a[indices]
    b = b[indices]
    return a, b
    # a, array([3, 4, 1, 2, 0])
    # b, array([8, 9, 6, 7, 5])


def PCA(ref2d):
    features = ref2d.T
    cov_matrix = np.cov(features)
    values, vectors = np.linalg.eig(cov_matrix)
    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))
    print(np.sum(explained_variances), '\n', explained_variances)


def pick_optimum_k(dataset):
    l = int(len(dataset) ** .5)
    if l % 2 == 0:
        return l + 1
    return l
