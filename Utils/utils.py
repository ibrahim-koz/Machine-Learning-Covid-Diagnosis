import random
import cv2
# Returns index of x in arr if present, else -1
import numpy as np
from Utils.distance_metrics import mahalanobis


def binary_search(arr, l, r, x):
    # Check base case
    if r >= l:

        mid = l + (r - l) // 2

        # If element is present at the middle itself
        if arr[mid] == x:
            return mid

            # If element is smaller than mid, then it
        # can only be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, l, mid - 1, x)

            # Else the element can only be present
        # in right subarray
        else:
            return binary_search(arr, mid + 1, r, x)

    else:
        # Element is not present in the array
        return -1


def split_train_test(dataset, split_ratio):
    train_sample = random.sample(range(0, len(dataset)), int(len(dataset) * 0.67))
    train_sample.sort()

    test_sample = []
    for i in range(0, len(dataset)):
        if binary_search(train_sample, 0, len(train_sample) - 1, i) == -1:
            test_sample.append(i)
    random.shuffle(test_sample)

    return train_sample, test_sample



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





