import sys
sys.path.append("..")
import numpy as np
from utils.BaseSVDD import BaseSVDD
from sklearn.decomposition import KernelPCA
import time

class openSVDD(BaseSVDD):
    def __init__(self,
                 C=0.9,
                 kernel='rbf',
                 degree=3,
                 gamma=None,
                 coef0=1,
                 display='on',
                 n_jobs=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.n_jobs = n_jobs
        self.display = display
        self.feature_means = ()
        self.feature_cell = ()
        self.unique_label = None
        self.class_num = None
        self.svdd = ()

    def get_features(self, features, labels):
        start_time = time.time()

        self.unique_label = set(labels)
        self.class_num = len(self.unique_label)
        # feature_means = np.empty([len(self.unique_label), features.shape[1]])
        for label in self.unique_label:
            label_index = [index for index, element in enumerate(labels) if element == label]
            feature = features[label_index, :]
            self.feature_cell += (feature,)
            feature_mean = np.mean(feature, axis=0)
            self.feature_means += (feature_mean,)

    def fit_openSVDD(self, features, labels):
        start_time = time.time()

        self.get_features(features, labels)
        for label in self.unique_label:
            feature_mean = self.feature_means[label]
            distances = np.linalg.norm(self.feature_means - feature_mean, axis=1)
            distances = np.where(distances == 0, np.inf, distances)
            min_value = np.min(distances)
            # 找到新最小值的索引
            min_index = np.unravel_index(np.argmin(distances), distances.shape)
            positive_feature = self.feature_cell[label]
            negative_feature = self.feature_cell[min_index[0]]
            svdd_feature = np.concatenate((positive_feature, negative_feature), axis=0)

            svdd_label = np.append(np.ones((len(positive_feature), 1), dtype=np.int64),
              -np.ones((len(negative_feature), 1), dtype=np.int64), axis=0)
            locals()['svdd_' + str(label)] = BaseSVDD(C=self.C, gamma=self.gamma, kernel=self.kernel, display=self.display,
                                 degree=self.degree, coef0=self.coef0, n_jobs=self.n_jobs)

            locals()['svdd_' + str(label)].fit(svdd_feature, svdd_label)
            self.svdd += (locals()['svdd_' + str(label)],)

    def predict(self, features, labels, threshold=None):
        radius_list = []
        correct = 0
        for i in range(len(self.svdd)):
            radius_list.append(self.svdd[i].radius)
            dis = self.svdd[i].get_distance(features)
            if i == 0:
                distance = dis
            else:
                distance = np.column_stack((distance, dis))

        # np.savez('./distance.npz',distance=distance)
        radius_array =np.array(radius_list).reshape(1, -1)
        norm_distance = distance / radius_array
        #求每一行的最小值
        min_values = norm_distance.min(axis=1)
        minmax = min_values.max()
        minmin = min_values.min()  # 求每一行最小值的列索引
        predict = np.array(np.argmin(norm_distance, axis=1)).squeeze()

        # indices = np.where(min_values > 6)
        if threshold == None:
            indices = np.where(min_values > 1)
            predict[indices[0]] = -1
        else:
            indices = np.where(min_values > threshold)

            # min_value_idx_list = np.zeros_like(predict, dtype=np.float32)
            # for i, min_value_idx in enumerate(predict):
            #     min_value_idx_list[i] = threshold[min_value_idx]
            #
            # threshold = min_value_idx_list
            # indices = np.where(np.array(min_values).squeeze() > threshold)[0]
            predict[indices[0]] = -1
        correct += (predict == labels).sum().item()
        accuracy = correct / predict.size

        return predict, min_values, accuracy, norm_distance














