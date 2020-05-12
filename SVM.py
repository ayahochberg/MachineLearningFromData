from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    partition_size = int(data.shape[0] * train_ratio)

    concatenated_data = permutation(concatenate((data, array([labels]).T), axis=1))

    shuffled_labels = concatenated_data[:, -1]
    shuffled_data = concatenated_data[:, :-1]

    train_data = shuffled_data[:partition_size]
    train_labels = shuffled_labels[:partition_size]

    test_data = shuffled_data[partition_size:]
    test_labels = shuffled_labels[partition_size:]

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """
    index_count = len(prediction)

    true_prediction = true_positive_count = false_positive_count = 0

    for i in range(index_count):
        if prediction[i] == labels[i]:
            true_prediction += 1
            if prediction[i] == 1:
                true_positive_count += 1
        elif labels[i] == 0:
            false_positive_count += 1

    actual_positive_events = sum(labels)
    actual_negative_events = len(labels) - actual_positive_events

    fpr = 0.0 if actual_negative_events == 0 else false_positive_count / actual_negative_events
    tpr = 0.0 if actual_positive_events == 0 else true_positive_count / actual_positive_events
    accuracy = true_prediction / index_count

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []
    folds_array_size = len(folds_array)

    for i in range(folds_array_size):
        test_features = folds_array.pop(0)
        train_features = concatenate(folds_array)

        test_labels = labels_array.pop(0)
        train_labels = concatenate(labels_array)

        clf.fit(train_features, train_labels)
        prediction = clf.predict(test_features)

        tp, fp, ac = get_stats(prediction, test_labels)

        tpr.append(tp)
        fpr.append(fp)
        accuracy.append(ac)

        folds_array.append(test_features)
        labels_array.append(test_labels)

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=(
                         {'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05},
                         {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params

    tpr_list = []
    fpr_list = []
    accuracy_list = []
    partitioned_data = array_split(data_array, folds_count)
    partitioned_labels = array_split(labels_array, folds_count)
    kernels_list_size = len(kernels_list)

    for i in range(kernels_list_size):
        c_val = SVM_DEFAULT_C
        degree_val = SVM_DEFAULT_DEGREE
        gamma_val = SVM_DEFAULT_GAMMA

        keys = list(kernel_params[i].keys())
        values = list(kernel_params[i].values())
        for j in range(len(keys)):
            key = keys[j]

            if key == 'gamma':
                gamma_val = values[j]
            elif key == 'degree':
                degree_val = values[j]
            elif key == 'C':
                c_val = values[j]

        clf = SVC(C=c_val, kernel=kernels_list[i], degree=degree_val, gamma=gamma_val)

        tpr, fpr, accuracy = get_k_fold_stats(partitioned_data, partitioned_labels, clf)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        accuracy_list.append(accuracy)

    svm_df['tpr'] = tpr_list
    svm_df['fpr'] = fpr_list
    svm_df['accuracy'] = accuracy_list

    return svm_df


def get_most_accurate_kernel(accuracy_list):
    """
    :return: integer representing the row number of the most accurate kernel
    """
    max_accuracy = max(accuracy_list)

    for i in range(len(accuracy_list)):
        if accuracy_list[i] == max_accuracy:
            best_kernel = i

    return best_kernel


def get_kernel_with_highest_score(score_list):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    max_score = max(score_list)

    for i in range(len(score_list)):
        if score_list[i] == max_score:
            best_kernel = i

    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x_values = df.fpr.tolist()
    y_values = df.tpr.tolist()

    b = (df['score'][get_kernel_with_highest_score]) / alpha_slope
    a = 1 / alpha_slope

    y_1 = [i * a for i in x_values] + b

    plt.title('ROC curve')
    plt.plot(x_values, y_1, color='red', lw=2)
    plt.scatter(x_values, y_values)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def evaluate_c_param(data_array, labels_array, folds_count, best_score_svm_params):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    c_values = []

    # Calc possible c values
    for i in [1, 0, -1, -2, -3, -4]:
        for j in [3, 2, 1]:
            c = ((10 ** i) * (j / 3))
            c_values.append(c)

    kernel_params = []

    best_score = best_score_svm_params.copy()
    for c in c_values:
        best_score['C'] = c
        kernel_params.append(best_score.copy())

    ker = list(best_score_svm_params.keys())

    kernel = 'poli' if ker[0] == 'degree' else 'rbf'
    kernels_list = []

    for i in range(len(c_values)):
        kernels_list.append(kernel)

    res = compare_svms(data_array, labels_array, folds_count, kernels_list, kernel_params)

    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels, best_kernel, best_kernel_params):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = best_kernel
    kernel_params = best_kernel_params

    clf = SVC(class_weight='balanced', kernel=kernel_type, **kernel_params)

    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    clf.fit(train_data, train_labels)
    check_labels = clf.predict(test_data)

    tpr, fpr, accuracy = get_stats(check_labels, test_labels)

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
