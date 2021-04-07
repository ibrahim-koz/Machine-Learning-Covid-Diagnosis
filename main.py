from Utils.filters import Gabor_process
from Utils.filters import Canny_edge
import numpy as np
import cv2
from Utils.utils import shuffle_together
from os import listdir
from os.path import isfile, join
from enum import Enum
from Utils.utils import image_resize
from Utils.utils import pick_optimum_k
from knn import KNN
from numba import njit

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


def test(p_covid_cases, p_normal_cases, p_viral_cases):
    global knn
    from sklearn.preprocessing import StandardScaler
    yCovid = np.full((len(p_covid_cases),), 0)
    yNormal = np.full((len(p_normal_cases),), 1)
    yViral = np.full((len(p_viral_cases),), 2)
    concatenated_matrix = np.concatenate((p_covid_cases, p_normal_cases, p_viral_cases), axis=0)
    cases_std = StandardScaler().fit_transform(concatenated_matrix)
    yCases = np.concatenate((yCovid, yNormal, yViral), axis=0)
    cases_std, yCases = shuffle_together(cases_std, yCases)
    # PCA(cases_std)

    from sklearn.decomposition import PCA
    pca = PCA(.98)

    from sklearn.model_selection import KFold
    split_num = 2
    kf = KFold(n_splits=split_num)

    correct_predictions = 0
    false_predictions = 0

    weighted_correct_predictions = 0
    weighted_false_predictions = 0


    for train_index, test_index in kf.split(cases_std):
        X_train, X_test = cases_std[train_index], cases_std[test_index]
        y_train, y_test = yCases[train_index], yCases[test_index]
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        knn = KNN(X_train, y_train, 6, list(range(len(X_train[0]))))

        for i in range(len(X_test)):
            if knn.get_knn(X_test[i]) == y_test[i]:
                correct_predictions += 1

        for i in range(len(X_test)):
            if knn.get_weighted_knn(X_test[i]) == y_test[i]:
                weighted_correct_predictions += 1

    print("LOOCV accuracy: ", correct_predictions / split_num)

    print("LOOCV weighted accuracy: ", weighted_correct_predictions / split_num)

import time
start_time = time.time()

covid_cases_gab, covid_cases_can = process_directory(r'train/COVID/')
normal_cases_gab, normal_cases_can = process_directory(r'train/NORMAL/')
viral_cases_gab, viral_cases_can = process_directory(r'train/Viral Pneumonia/')

covid_cases = np.concatenate((covid_cases_gab, covid_cases_can), axis=1)
normal_cases = np.concatenate((normal_cases_gab, normal_cases_can), axis=1)
viral_cases = np.concatenate((viral_cases_gab, viral_cases_can), axis=1)

print("THE RESULTS GOT FROM GABOR")
test(covid_cases_gab, normal_cases_gab, viral_cases_gab)
print("THIS IS USED TO DISTINGUISH THE TEST RESULTS")
print("THE RESULTS GOT FROM CANNY")
test(covid_cases_can, normal_cases_can, viral_cases_can)
print("THIS IS USED TO DISTINGUISH THE TEST RESULTS")
print("THE RESULTS GOT FROM THE WAY IN WHICH THEY ARE COMBINED")
test(covid_cases, normal_cases, viral_cases)

print("--- %s seconds ---" % (time.time() - start_time))

print("end")
