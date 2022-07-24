import pickle
import tensorflow
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.svm import SVR
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import linear_model
import warnings

def get_class(x):
    classes = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.32, 0.4, 0.5, 0.63, 0.8, 1]
    diff = [(abs(c-x), c) for c in classes]
    diff.sort()
    abs_value, value = diff[0]
    return value

def evaluate_regressor():
    warnings.filterwarnings('ignore')
    with open('D:/AN4/Licenta/Out/patients_info.txt', 'rb') as dataframe_file:
        patient_info = pickle.load(dataframe_file)

    with open('D:/AN4/Licenta/Out/tensors_normalized.txt', 'rb') as normalized_file:
        tensors_normalized = pickle.load(normalized_file)

    tensors = [t.numpy() for t in tensors_normalized]

    nr_patients = len(patient_info)

    # get number of distinct patients
    distinct_patient = np.array(list(set([p[0] for p in patient_info])))

    # get 70 distinct patients for train and 10 for test
    np.random.seed(10)
    patient_train = np.random.choice(distinct_patient, size = 70, replace=False).tolist()
    patient_test = [p for p in distinct_patient if p not in patient_train]

    # get indexes for volumetric images corresponding to train and test patients
    index_train=[i for i in range(nr_patients) if patient_info[i][0]  in patient_train and patient_info[i][4]!='cps']
    index_test=[i for i in range(nr_patients) if patient_info[i][0] in patient_test and patient_info[i][4]!='cps']

    print("Selected ", len(index_train), "examples for train.")
    print("Selected ", len(index_test), "examples for test.")

    y_train=[patient_info[i][3] for i in index_train]
    y_test=[patient_info[i][3] for i in index_test]

    # compute features for SVR as average value per vector feature
    avg = [t.sum(axis = 0)/len(t) for t in tensors]

    X_train=[avg[i] for i in index_train]
    X_test=[avg[i] for i in index_test]

    svm_model = SVR(kernel = 'linear')
    svm_model.fit(X_train, y_train)
    svm_model.score(X_train, y_train)
    svm_model.score(X_test, y_test)

    # y_pred = svm_model.predict(X_train)
    # print(f'RMSE train {math.sqrt(mean_squared_error(y_train, y_pred))}')
    # print(f'MAE train {mean_absolute_error(y_train, y_pred)}')

    y_pred = svm_model.predict(X_test)
    print(f'RMSE test {math.sqrt(mean_squared_error(y_test, y_pred))}')
    print(f'MAE test {mean_absolute_error(y_test, y_pred)}')

    classes = ['0', '0.01', '0.025', '0.05', '0.1', '0.2', '0.32', '0.4', '0.5', '0.63', '0.8', '1']
    y_pred_classes = [get_class(y_p) for y_p in y_pred]

    print(classification_report([str(y) for y in y_test], [str(y) for y in y_pred_classes]))
    cm = confusion_matrix([str(y) for y in y_test], [str(y) for y in y_pred_classes], labels=classes)
    print(cm)
    sns.heatmap(cm, cmap="Blues", annot=True)
