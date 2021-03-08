from knn import KNN
from Utils.filters import Gabor_process
from Utils.filters import Canny_edge
import numpy as np
import cv2
from Utils.utils import split_train_test
from os import listdir
from os.path import isfile, join
from enum import Enum
from Utils.utils import image_resize


class Label(Enum):
    COVID = 0
    NORMAL = 1
    VIRAL = 2


def process_data(file_path):
    # Read image
    img = cv2.imread(file_path)
    img = image_resize(img, 32, 32)

    img32 = img.astype(np.float32)
    # gabor process
    gab_out = Gabor_process(img32)

    # canny edge process
    can_out = Canny_edge(img)

    return gab_out.flatten(), can_out.flatten()


# TODO: dimensionality reduction will be implemented

def train(out):
    training_set, test_set = split_train_test(out, 0.67)
    knn = KNN(out, training_set, 3, [0, 1, 2])
    results = [knn.get_knn(out[test_set[i]]) for i in range(len(test_set))]


class Part(Enum):
    X = 0
    Y = 1


def process_directory(dir_path):
    whole_dataset = []
    for f in listdir(dir_path):
        file_path = join(dir_path + f)
        gab_out, can_out = process_data(file_path)
        whole_dataset.append(gab_out)
    return whole_dataset


covid_cases = process_directory(r'train/COVID_SAMPLE/')
normal_cases = process_directory(r'train/NORMAL_SAMPLE/')
viral_cases = process_directory(r'train/VIRAL_SAMPLE/')


def PCA(ref2d):
    features = ref2d.T
    cov_matrix = np.cov(features)
    values, vectors = np.linalg.eig(cov_matrix)
    explained_variances = []
    for i in range(len(values)):
        explained_variances.append(values[i] / np.sum(values))
    print(np.sum(explained_variances), '\n', explained_variances)


from sklearn.preprocessing import StandardScaler

covid_cases_std = StandardScaler().fit_transform(covid_cases)
normal_cases_std = StandardScaler().fit_transform(normal_cases)
viral_cases_std = StandardScaler().fit_transform(viral_cases)

yCovid = np.full((108,), Label.COVID)
yNormal = np.full((108,), Label.NORMAL)
yViral = np.full((108,), Label.VIRAL)

cases_std = np.concatenate((covid_cases_std, normal_cases_std, viral_cases_std), axis=0)
yCases = np.concatenate((yCovid, yNormal, yViral), axis=0)

from Utils.utils import shuffle_together
cases_std, yCases = shuffle_together(cases_std, yCases)

PCA(cases_std)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(cases_std, yCases, test_size=0.2)



print("end")
