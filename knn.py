import operator
from Utils.utils import euclidean_distance


class KNN:
    def __init__(self, data, training_set, k, attributes):
        self.data = data
        self.training_set = training_set
        self.k = k
        self.attributes = attributes

    def get_knn(self, test_instance):
        distances = []
        for i in self.training_set:
            distances.append([euclidean_distance(self.data[i], test_instance, self.attributes), i])
        distances.sort(key=lambda x: x[0], reverse=True)

        most_k = distances[-self.k:]

        votes = dict()
        for i in range(len(most_k)):
            if self.data[most_k[i][1]][4] in votes:
                votes[self.data[most_k[i][1]][4]] += 1
            else:
                votes[self.data[most_k[i][1]][4]] = 1
        return max(votes.items(), key=operator.itemgetter(1))[0]


