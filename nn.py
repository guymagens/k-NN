import numpy
import heapq
import numpy.random
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

def KNearestNeighbour(images, labels, query_image, k):
    nearests = []

    for i in range(len(images)):
        dist = -numpy.linalg.norm(query_image - images[i])
        if len(nearests) < k:
            heapq.heappush(nearests, (dist, labels[i]))
        else:
            if dist > nearests[0][0]:
                heapq.heappushpop(nearests, (dist, labels[i]))
    digit_count = [[0, i] for i in range(10)]
    for nearest in nearests:
        digit_count[int(nearest[1])][0] += 1

    return max(digit_count, key=lambda x: x[0])[1]

def CheckKNN(train, train_labels, test, test_labels, N, k):
    correct = 0
    for i in range(len(test)):
        correct += KNearestNeighbour(train[:N], train_labels, test[i], k) == int(test_labels[i])
    acc = correct/ len(test)
    print(f'For k = {k}, N={N} precentage: {acc}')
    return acc

def main():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']
    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]
    x = [k for k in range(1,101)]
    y = [CheckKNN(train, train_labels, test, test_labels, 1000, k) for k in x]
    plt.title("Prediction accuracy as a function of k")
    plt.xlabel("k")
    plt.ylabel("Prediction accuracy")
    plt.subplot(2,1,1)
    plt.plot(x, y)

    x = [N for N in range(100,5001,100)]
    y = [CheckKNN(train, train_labels, test, test_labels, N, 1) for N in x]
    plt.title("Prediction accuracy as a function of k")
    plt.xlabel("N")
    plt.ylabel("Prediction accuracy")
    plt.subplot(2,1,2)
    plt.plot(x, y)
    plt.show()
