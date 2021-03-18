from Utils.filters import Gabor_process
from Utils.filters import Canny_edge
import numpy as np
import cv2
from Utils.utils import split_train_test, shuffle_together
from os import listdir
from os.path import isfile, join
from enum import Enum
from Utils.utils import image_resize
from Utils.utils import pick_optimum_k
from knn import KNN


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
    gab_whole_dataset = []
    can_whole_dataset = []
    for f in listdir(dir_path):
        file_path = join(dir_path + f)
        gab_out, can_out = process_data(file_path)
        if len(gab_out) == 1024:
            gab_whole_dataset.append(gab_out)
            can_whole_dataset.append(can_out)
    return gab_whole_dataset, can_whole_dataset


def test(covid_cases, normal_cases, viral_cases):
    global knn
    from sklearn.preprocessing import StandardScaler
    yCovid = np.full((len(covid_cases),), 0)
    yNormal = np.full((len(normal_cases),), 1)
    yViral = np.full((len(viral_cases),), 2)
    concatenated_matrix = np.concatenate((covid_cases, normal_cases, viral_cases), axis=0)
    cases_std = StandardScaler().fit_transform(concatenated_matrix)
    yCases = np.concatenate((yCovid, yNormal, yViral), axis=0)
    cases_std, yCases = shuffle_together(cases_std, yCases)
    # PCA(cases_std)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(cases_std, yCases, test_size=0.2)

    from sklearn.decomposition import PCA

    pca = PCA(.95)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    knn = KNN(X_train, y_train, pick_optimum_k(X_train), list(range(len(X_train[0]))))
    y_pred = []
    for i in range(len(X_test)):
        y_pred.append(knn.get_knn(X_test[i]))
    y_pred = np.array(y_pred)
    weighted_y_pred = knn.get_weighted_knn(X_test)

    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    weighted_conf_matrix = confusion_matrix(y_test, weighted_y_pred)
    print(weighted_conf_matrix)

covid_cases_gab, covid_cases_can = process_directory(r'train/COVID_SAMPLE30/')
normal_cases_gab, normal_cases_can = process_directory(r'train/NORMAL_SAMPLE30/')
viral_cases_gab, viral_cases_can = process_directory(r'train/VIRAL_SAMPLE30/')

covid_cases = np.concatenate((covid_cases_gab, covid_cases_can), axis=1)
normal_cases = np.concatenate((normal_cases_gab, normal_cases_can), axis=1)
viral_cases = np.concatenate((viral_cases_gab, viral_cases_can), axis=1)

# test(covid_cases_gab, normal_cases_gab, viral_cases_gab)
# test(covid_cases_can, normal_cases_can, viral_cases_can)
test(covid_cases, normal_cases, viral_cases)
print("end")
