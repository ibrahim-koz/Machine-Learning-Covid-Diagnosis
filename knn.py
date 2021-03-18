import operator

from Utils.distance_metrics import euclidean_distance
from Utils.distance_metrics import mahalanobis


class KNN:
    def __init__(self, x_train, y_train, k, attributes):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.attributes = attributes

    def find_distances(self, test_instance):
        distances = []
        for i in range(len(self.x_train)):
            distances.append([euclidean_distance(self.x_train[i], test_instance, self.attributes), i])
        return distances

    def get_knn(self, test_instance):
        distances = self.find_distances(test_instance)
        return self.decide(distances)

    def get_weighted_knn(self, test_instance):
        distances = self.find_distances(test_instance)
        return self.decide(distances, kernel_function=lambda x: 1/(x + 0.0000000001))

    def decide(self, distances, kernel_function=lambda x: 1):
        distances.sort(key=lambda x: x[0], reverse=True)
        most_k = distances[-self.k:]
        votes = dict()
        for i in range(len(most_k)):
            if self.y_train[most_k[i][1]] in votes:
                votes[self.y_train[most_k[i][1]]] += kernel_function(most_k[i][0])
            else:
                votes[self.y_train[most_k[i][1]]] = kernel_function(most_k[i][0])
        return max(votes.items(), key=operator.itemgetter(1))[0]
